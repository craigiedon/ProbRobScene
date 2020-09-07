"""Workspaces."""

from scenic3d.core.distributions import needsSampling
from scenic3d.core.geometry import min_and_max
from scenic3d.core.regions import Region, everywhere
from scenic3d.core.utils import RuntimeParseError
import numpy as np


class Workspace(Region):
    """A workspace describing the fixed world of a scenario"""

    def __init__(self, region=everywhere):
        if needsSampling(region):
            raise RuntimeParseError('workspace region must be fixed')
        super().__init__('workspace', orientation=region.orientation)
        self.region = region

    def show_3d(self, ax):
        aabb = self.region.getAABB()  # TODO: Come up with a 3d-version of this

        min_coords, max_coords = aabb
        total_min, total_max = np.min(min_coords), np.max(max_coords)

        ax.set_xlim(total_min, total_max)
        ax.set_ylim(total_min, total_max)
        ax.set_zlim(total_min, total_max)

    def show(self, plt):
        """Render a schematic of the workspace for debugging"""
        try:
            aabb = self.region.getAABB()
        except NotImplementedError:  # unbounded Regions don't support this
            return
        ((xmin, ymin), (xmax, ymax)) = aabb
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

    def zoomAround(self, plt, objects, expansion=2):
        """Zoom the schematic around the specified objects"""
        positions = (self.scenicToSchematicCoords(obj.position) for obj in objects)
        x, y = zip(*positions)
        minx, maxx = min_and_max(x)
        miny, maxy = min_and_max(y)
        sx = expansion * (maxx - minx)
        sy = expansion * (maxy - miny)
        s = max(sx, sy, self.minimumZoomSize) / 2.0
        s += max(max(obj.width, obj.height) for obj in objects)  # TODO improve
        cx = (maxx + minx) / 2.0
        cy = (maxy + miny) / 2.0
        plt.xlim(cx - s, cx + s)
        plt.ylim(cy - s, cy + s)

    @property
    def minimumZoomSize(self):
        return 0

    def scenicToSchematicCoords(self, coords):
        """Convert Scenic coordinates to those used for schematic rendering."""
        return coords

    def uniformPointInner(self):
        return self.region.uniformPointInner()

    def intersect(self, other, triedReversed=False):
        return self.region.intersect(other, triedReversed)

    def containsPoint(self, point):
        return self.region.containsPoint(point)

    def containsObject(self, obj):
        return self.region.containsObject(obj)

    def getAABB(self):
        return self.region.getAABB()

    def __str__(self):
        return f'<Workspace on {self.region}>'

    def __eq__(self, other):
        if type(other) is not Workspace:
            return NotImplemented
        return other.region == self.region

    def __hash__(self):
        return hash(self.region)
