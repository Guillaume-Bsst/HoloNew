"""Point clouds + correspondence (build-once): the SMPL surface and each object surface as
``PointCloud`` (sparse skinning), plus the SMPL<->robot ``CorrespondenceTable``. Logic in the
submodules; the human cloud and the correspondence share one ``SurfaceSampling`` identity."""
from .human import HumanCloudBuilder, bake_skinned_cloud, build_human_cloud
from .objects import ObjectCloudBuilder, assemble_rigid_cloud, build_object_cloud, sample_object_surface
from .sampling import SurfaceSampling, sampling_id

__all__ = ["bake_skinned_cloud", "build_human_cloud", "HumanCloudBuilder",
           "sample_object_surface", "assemble_rigid_cloud", "build_object_cloud", "ObjectCloudBuilder",
           "SurfaceSampling", "sampling_id"]
