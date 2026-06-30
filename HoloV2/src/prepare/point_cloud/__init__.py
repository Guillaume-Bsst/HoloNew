"""Nuages de points + correspondance (build-once) : la surface SMPL et chaque surface d'objet comme
``PointCloud`` (skinning creux), plus la table de correspondance SMPL<->robot ``CorrespondenceTable``.
La logique réside dans les sous-modules ; le nuage humain et la correspondance partagent une
identité ``SurfaceSampling`` unique."""
from .human import HumanCloudBuilder, bake_skinned_cloud, build_human_cloud
from .objects import ObjectCloudBuilder, assemble_rigid_cloud, build_object_cloud, sample_object_surface
from .sampling import SurfaceSampling, sampling_id

__all__ = ["bake_skinned_cloud", "build_human_cloud", "HumanCloudBuilder",
           "sample_object_surface", "assemble_rigid_cloud", "build_object_cloud", "ObjectCloudBuilder",
           "SurfaceSampling", "sampling_id"]
