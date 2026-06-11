# src/test_pipe_retargeting/test_pipe_retargeting/fields/backends/coal.py
"""Coal-direct engine: BVH build/refit and surface field variants."""
from __future__ import annotations

import numpy as np

from ..contact_field import ContactField, _contains, _probe_distance


def build_bvh(verts: np.ndarray, faces: np.ndarray):
    """Coal BVH (OBBRSS) from a triangle mesh."""
    import coal
    bvh = coal.BVHModelOBBRSS()
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    bvh.beginModel(len(f), len(v))
    bvh.addVertices(v)
    bvh.addTriangles(f)
    bvh.endModel()
    return bvh


def refit_bvh(bvh, verts: np.ndarray) -> None:
    """Refit a Coal BVH hierarchy with new vertex positions (topology must match).
    Faster than a full rebuild for deforming meshes like the human body."""
    import coal
    v = np.asarray(verts, dtype=np.float64)
    bvh.beginReplaceModel()
    sv = coal.StdVec_Vec3s()
    sv.extend(v)
    bvh.replaceSubModel(sv)
    bvh.endReplaceModel(True, True)


def surface_field(
    probe_pts: np.ndarray,
    target_bvh,          # coal.BVHModelOBBRSS from build_bvh()
    target_mesh,         # trimesh.Trimesh of the same surface, for the inside test
    margin: float,       # metres; a probe is active iff signed_dist < margin (strict)
) -> ContactField:
    """Signed surface field from probe_pts to target_bvh (both in the same frame).

    Per probe: distance < 0 means penetrating; distance >= 0 means outside (clamped
    to +margin when inactive). `active` is `signed_dist < margin` — strictly less,
    so a penetrating probe is always active and a probe exactly at +margin is not.
    `witness` is always the nearest surface point regardless of `active`.
    """
    import coal
    pts = np.asarray(probe_pts, dtype=np.float64)
    n = len(pts)
    dist = np.empty(n, dtype=np.float64)
    direction = np.zeros((n, 3), dtype=np.float64)
    witness = np.zeros((n, 3), dtype=np.float64)

    sphere = coal.Sphere(0.0)
    tf_mesh = coal.Transform3s()
    tf_pt = coal.Transform3s()
    req = coal.DistanceRequest()

    for i in range(n):
        dist[i], witness[i], direction[i] = _probe_distance(
            pts[i], target_bvh, sphere, tf_pt, tf_mesh, req)

    inside = _contains(target_mesh, pts)
    signed = np.where(inside, -dist, dist)

    active = signed < margin
    out_dist = np.where(active, signed, margin).astype(np.float32)
    out_dir = np.where(active[:, None], direction, 0.0).astype(np.float32)
    return ContactField(
        distance=out_dist, direction=out_dir,
        witness=witness.astype(np.float32), active=active,
    )


def surface_field_batched(
    probe_pts: np.ndarray,
    target_bvh,          # coal.BVHModelOBBRSS from build_bvh()
    target_mesh,         # trimesh.Trimesh of the same surface, for inside test
    margin: float,       # metres; same contract as surface_field
    resolution: float | None = None,
    tree: "cKDTree | None" = None,
    inside: np.ndarray | None = None,
) -> ContactField:
    """Signed surface field — same contract as surface_field but faster for large probe sets.

    Note: unlike surface_field (which queries every probe), `witness` is only filled for
    active probes here; inactive probes keep a zero witness. distance/direction/active
    follow the same contract as surface_field.

    Uses Coal's octree-vs-BVH broadphase to gate which probes are within `margin` of
    the surface in one call, then runs the exact per-probe Coal distance query only on
    that active subset.  Probes deeper than the octree resolution inside the mesh are
    caught by a trimesh contains() pre-pass.  Far probes (> margin from surface) skip
    the per-probe loop entirely, giving a roughly N_active/N speedup.

    The two-stage approach (octree gate + exact per-probe distance on the active subset)
    is necessary because Coal's BVH is a surface structure: for a probe fully enclosed
    inside a mesh the octree voxel may not intersect any triangle, so it would be missed
    if we relied on the octree alone.

    Args:
        probe_pts: (N, 3) probe positions.
        target_bvh: coal.BVHModelOBBRSS built by build_bvh().
        target_mesh: trimesh.Trimesh of the same surface (for the inside test).
        margin: probes with signed_dist < margin are active; inactive probes are
            clamped to +margin.
        resolution: octree voxel size passed to coal.makeOctree.  Defaults to
            margin / 2 — small enough that a probe at the boundary of the active
            band is covered by its voxel, large enough to keep the octree compact.
        tree: optional precomputed scipy.spatial.cKDTree(probe_pts).
        inside: optional precomputed boolean array (N,) from trimesh.contains(probe_pts).
    """
    import coal
    from scipy.spatial import cKDTree

    pts = np.asarray(probe_pts, dtype=np.float64)
    n = len(pts)
    if resolution is None:
        resolution = float(margin) / 2.0

    # --- Stage 1: detect inside probes (always active regardless of octree) ---
    if inside is None:
        inside = _contains(target_mesh, pts)

    # --- Stage 2: octree broadphase to find outside-but-near probes ---
    octree = coal.makeOctree(pts, float(resolution))
    req = coal.CollisionRequest()
    req.security_margin = float(margin)
    # Higher cap to avoid truncation in dense contact regions.
    req.num_max_contacts = max(10000, n * 200)
    res = coal.CollisionResult()
    tf = coal.Transform3s()
    coal.collide(octree, tf, target_bvh, tf, req, res)

    # Coal silently truncates the contact list at num_max_contacts; if hit, near-
    # outside probes could be dropped from the gate (the inside pre-pass only saves
    # penetrating ones). Warn so the cap can be raised rather than fail silently.
    if res.numContacts() >= req.num_max_contacts:
        import warnings
        warnings.warn(
            f"surface_field_batched: contact list may be truncated "
            f"({res.numContacts()} >= cap {req.num_max_contacts}); "
            "raise num_max_contacts or lower probe/mesh density.",
            RuntimeWarning, stacklevel=2,
        )

    # getNearestPoint1() is a point on the octree voxel, not the probe itself, so the
    # nearest-probe lookup is approximate; the stage-3 exact recheck makes any
    # mis-assignment harmless (it can only add a false positive to the gate).
    # The collide call can return tens of thousands of contacts; map them all to their
    # nearest probe in ONE batched cKDTree query rather than a per-contact Python loop
    # (the per-contact loop dominated runtime).
    oct_hit = np.zeros(n, dtype=bool)
    contacts = res.getContacts()
    if contacts:
        p1 = np.array([c.getNearestPoint1() for c in contacts], dtype=np.float64)
        if tree is None:
            tree = cKDTree(pts)
        _, idx = tree.query(p1)
        oct_hit[idx] = True

    # Union: a probe is a gate candidate if it is inside OR touched by the octree.
    gate = inside | oct_hit

    # --- Stage 3: exact per-probe Coal distance on the gated subset only ---
    dist = np.full(n, float(margin), dtype=np.float64)
    direction = np.zeros((n, 3), dtype=np.float64)
    witness = np.zeros((n, 3), dtype=np.float64)
    active = np.zeros(n, dtype=bool)

    if gate.any():
        sphere_pt = coal.Sphere(0.0)
        tf_mesh = coal.Transform3s()
        tf_pt = coal.Transform3s()
        req_d = coal.DistanceRequest()

        for i in np.where(gate)[0]:
            unsigned_d, witness[i], direction[i] = _probe_distance(
                pts[i], target_bvh, sphere_pt, tf_pt, tf_mesh, req_d)
            signed = -unsigned_d if inside[i] else unsigned_d
            dist[i] = signed
            if signed < margin:
                active[i] = True

    out_dist = np.where(active, dist, float(margin)).astype(np.float32)
    out_dir = np.where(active[:, None], direction, 0.0).astype(np.float32)
    return ContactField(
        distance=out_dist, direction=out_dir,
        witness=witness.astype(np.float32), active=active,
    )


def surface_field_kdtree(
    probe_pts: np.ndarray,
    target_bvh,          # coal.BVHModelOBBRSS from build_bvh()
    target_mesh,         # trimesh.Trimesh of the same surface, for inside test + gate
    margin: float,       # metres; same contract as surface_field
    resolution: float | None = None,   # accepted for drop-in compat; unused here
    tree: "cKDTree | None" = None,      # accepted for drop-in compat; unused here
    inside: np.ndarray | None = None,
    vert_tree: "cKDTree | None" = None,  # optional cached cKDTree(target_mesh.vertices)
    edge_slack: float | None = None,     # optional cached max edge length of target_mesh
) -> ContactField:
    """Signed surface field — same contract as surface_field_batched, but the
    broadphase gate uses a cKDTree nearest-vertex query instead of Coal's
    octree+collide pass (which dominated runtime).

    Gate (correct superset of the active set): a probe can only be active if its
    nearest *surface* point is within `margin`. That surface point lies on a
    triangle whose nearest *vertex* is at most one edge-length away, so

        nearest_vertex_distance <= margin + max_edge_length

    holds for every active probe. Gating on that bound therefore never drops an
    active probe; the exact per-probe Coal recheck (stage 3) removes the few
    false positives the slack lets in. Deeply interior probes (whose nearest
    vertex may be far) are caught by the trimesh contains() pre-pass, exactly as
    in surface_field_batched.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(probe_pts, dtype=np.float64)
    n = len(pts)
    verts = np.asarray(target_mesh.vertices, dtype=np.float64)

    # --- Stage 1: detect inside probes (always active regardless of the gate) ---
    if inside is None:
        inside = _contains(target_mesh, pts)

    # --- Stage 2: cKDTree nearest-vertex broadphase gate ---
    if vert_tree is None:
        vert_tree = cKDTree(verts)
    if edge_slack is None:
        edge_slack = float(target_mesh.edges_unique_length.max())
    d_near, _ = vert_tree.query(pts)
    gate = inside | (d_near < float(margin) + edge_slack)

    # --- Stage 3: exact per-probe Coal distance on the gated subset only ---
    dist = np.full(n, float(margin), dtype=np.float64)
    direction = np.zeros((n, 3), dtype=np.float64)
    witness = np.zeros((n, 3), dtype=np.float64)
    active = np.zeros(n, dtype=bool)

    if gate.any():
        import coal
        sphere_pt = coal.Sphere(0.0)
        tf_mesh = coal.Transform3s()
        tf_pt = coal.Transform3s()
        req_d = coal.DistanceRequest()

        for i in np.where(gate)[0]:
            unsigned_d, witness[i], direction[i] = _probe_distance(
                pts[i], target_bvh, sphere_pt, tf_pt, tf_mesh, req_d)
            signed = -unsigned_d if inside[i] else unsigned_d
            dist[i] = signed
            if signed < margin:
                active[i] = True

    out_dist = np.where(active, dist, float(margin)).astype(np.float32)
    out_dir = np.where(active[:, None], direction, 0.0).astype(np.float32)
    return ContactField(
        distance=out_dist, direction=out_dir,
        witness=witness.astype(np.float32), active=active,
    )


def surface_distance_torch(
    probe_pts: "torch.Tensor",
    target_bvh,
    target_mesh,
    margin: float,
) -> "torch.Tensor":
    """Differentiable signed distance: value from Coal, gradient = contact normal.

    probe_pts: (N, 3) torch tensor. Returns (N,) signed distances with grad wired to
    the probe positions via the analytic d(dist)/d(p) = contact normal (direction,
    surface -> probe), zeroed for inactive probes.
    """
    import torch

    # Defined per-call so the closure captures target_bvh/target_mesh/margin, which
    # are non-tensor Coal objects and cannot be autograd.Function inputs.
    class _SurfaceDistance(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pts):
            # Performance: use the cKDTree-gated implementation for large probe sets.
            # Small sets (like in finite-difference tests) are safer/faster with the slow path.
            p_np = pts.detach().cpu().numpy()
            if len(p_np) > 128:
                f = surface_field_kdtree(p_np, target_bvh, target_mesh, margin)
            else:
                f = surface_field(p_np, target_bvh, target_mesh, margin)
            ctx.save_for_backward(
                torch.from_numpy(np.ascontiguousarray(f.direction)).to(pts),
                torch.from_numpy(np.ascontiguousarray(f.active)).to(pts.device),
            )
            return torch.from_numpy(np.ascontiguousarray(f.distance)).to(pts)

        @staticmethod
        def backward(ctx, grad_out):
            direction, active = ctx.saved_tensors
            grad_p = grad_out[:, None] * direction
            grad_p = grad_p * active[:, None].to(grad_p)
            return grad_p

    return _SurfaceDistance.apply(probe_pts)
