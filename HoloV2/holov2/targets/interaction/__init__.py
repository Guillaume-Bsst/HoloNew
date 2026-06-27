"""Interaction targets (online, per frame): pose the clouds, evaluate the channels, transport the
human field onto the robot. Pure ops in the submodules; ``pose_cloud`` is shared by every cloud."""
from .pointclouds import pose_cloud

__all__ = ["pose_cloud"]
