"""Pruning parts of the sample space which violate requirements."""

import math
import time

import scenic3d.core.regions as regions
from scenic3d.core.distributions import (Samplable, MethodDistribution, OperatorDistribution,
                                         support_interval, underlying_function)
from scenic3d.core.geometry import normalize_angle
from scenic3d.core.utils import InvalidScenarioError
from scenic3d.core.vectors import VectorField, PolygonalVectorField, VectorMethodDistribution
from scenic3d.syntax.relations import RelativeHeadingRelation, DistanceRelation


### Utilities

def currentPropValue(obj, prop):
    """Get the current value of an object's property, taking into account prior pruning."""
    value = getattr(obj, prop)
    return value._conditioned if isinstance(value, Samplable) else value


def isMethodCall(thing, method):
    """Match calls to a given method, taking into account distribution decorators."""
    if not isinstance(thing, (MethodDistribution, VectorMethodDistribution)):
        return False
    return thing.method is underlying_function(method)


# def match_in_region(position):
#     """Match uniform samples from a Region, returning the Region if any."""
#     if isinstance(position, regions.PointInRegionDistribution):
#         reg = position.region
#         # if isinstance(reg, Workspace):
#         #     reg = reg.region
#         # return reg
#     return None


def matchPolygonalField(heading, position):
    """Match headings defined by a PolygonalVectorField at the given position.

    Matches headings exactly equal to a PolygonalVectorField, or offset by a
    bounded disturbance. Returns a triplet consisting of the matched field if
    any, together with lower/upper bounds on the disturbance.
    """
    if (isMethodCall(heading, VectorField.__getitem__)
            and isinstance(heading.object, PolygonalVectorField)
            and heading.arguments == (position.toVector(),)):
        return heading.object, 0, 0
    elif isinstance(heading, OperatorDistribution) and heading.operator in ('__add__', '__radd__'):
        field, lower, upper = matchPolygonalField(heading.object, position)
        if field is not None:
            assert len(heading.operands) == 1
            offset = heading.operands[0]
            ol, oh = support_interval(offset)
            if ol is not None and oh is not None:
                return field, lower + ol, upper + oh
    return None, 0, 0


def prune(scenario, verbosity=1):
    if verbosity >= 1:
        print('  Pruning scenario...')
        start_time = time.time()

    prune_containment(scenario, verbosity)
    # prune_relative_heading(scenario, verbosity)

    if verbosity >= 1:
        total_time = time.time() - start_time
        print(f'  Pruned scenario in {total_time:.4g} seconds.')


def prune_containment(scenario, verbosity):
    prunable = [x for x in scenario.objects
                if isinstance(x.position, regions.PointInRegionDistribution)
                and isinstance(x.position.region, regions.Intersect)]
    for obj in prunable:
        base = obj.position.region
        container = scenario.workspace
        new_base = base.intersect(container)

        if isinstance(new_base, regions.EmptyRegion):
            raise InvalidScenarioError(f'Object {obj} does not intersect with {container}')

        # TODO: Calculate volume of convex poly? Probs not possible with infinite vol halfspaces...

        # TODO: if we can lower bound the radius, erode the container
        # if (minRadius := supportInterval(obj.inradius)[0]) is not None:
        #     container = erode(container, minRadius)

        new_pos = regions.PointInRegionDistribution(new_base)
        obj.position.condition_to(new_pos)


def prune_relative_heading(scenario, verbosity):
    """Prune based on requirements bounding the relative heading of an Object.

    Specifically, if an object O is:

        * positioned uniformly within a polygonal region B;
        * aligned to a polygonal vector field F (up to a bounded offset);

    and another object O' is:

        * aligned to a polygonal vector field F' (up to a bounded offset);
        * at most some finite maximum distance from O;
        * required to have relative heading within a bounded offset of that of O;

    then we can instead position O uniformly in the subset of B intersecting the cells
    of F which satisfy the relative heading requirements w.r.t. some cell of F' which
    is within the distance bound.
    """
    # Check which objects are (approximately) aligned to polygonal vector fields
    fields = {}
    for obj in scenario.objects:
        field, offset_l, offset_r = matchPolygonalField(obj.heading, obj.position)
        if field is not None:
            fields[obj] = (field, offset_l, offset_r)
    # Check for relative heading relations among such objects
    for obj, (field, offset_l, offset_r) in fields.items():
        position = currentPropValue(obj, 'position')
        base = match_in_region(position)
        if base is None:  # obj must be positioned uniformly in a Region
            continue
        basePoly = regions.toPolygon(base)
        if basePoly is None:  # the Region must be polygonal
            continue
        newBasePoly = basePoly
        for rel in obj._relations:
            if isinstance(rel, RelativeHeadingRelation) and rel.target in fields:
                tField, tOffsetL, tOffsetR = fields[rel.target]
                maxDist = max_distance_between(scenario, obj, rel.target)
                if maxDist == float('inf'):  # the distance between the objects must be bounded
                    continue
                feasible = feasibleRHPolygon(field, offset_l, offset_r,
                                             tField, tOffsetL, tOffsetR,
                                             rel.lower, rel.upper, maxDist)
                if feasible is None:  # the RH bounds may be too weak to restrict the space
                    continue
                try:
                    pruned = newBasePoly & feasible
                except shapely.geos.TopologicalError:  # TODO how can we prevent these??
                    pruned = newBasePoly & feasible.buffer(0.1, cap_style=2)
                if verbosity >= 1:
                    percent = 100 * (1.0 - (pruned.area / newBasePoly.area))
                    print(f'    Relative heading constraint pruned {percent:.1f}% of space.')
                newBasePoly = pruned
        if newBasePoly is not basePoly:
            newBase = regions.PolygonalRegion(polygon=newBasePoly,
                                              orientation=base.orientation)
            newPos = regions.PointInRegionDistribution(newBase)
            obj.position.condition_to(newPos)


def max_distance_between(scenario, obj, target):
    """Upper bound the distance between the given Objects."""
    # Check for any distance bounds implied by user-specified requirements
    reqDist = float('inf')
    for rel in obj._relations:
        if isinstance(rel, DistanceRelation) and rel.target is target:
            if rel.upper < reqDist:
                reqDist = rel.upper

    return reqDist


def visibilityBound(obj, target):
    """Upper bound the distance from an Object to another it can see."""
    # Upper bound on visible distance is a sum of several terms:
    # 1. obj.visibleDistance
    _, maxVisibleDistance = support_interval(obj.visibleDistance)
    if maxVisibleDistance is None:
        return None
    # 2. distance from obj's center to its camera
    _, maxCameraX = support_interval(obj.cameraOffset.x)
    _, maxCameraY = support_interval(obj.cameraOffset.y)
    if maxCameraX is None or maxCameraY is None:
        return None
    maxVisibleDistance += math.hypot(maxCameraX, maxCameraY)
    # 3. radius of target
    _, maxRadius = support_interval(target.radius)
    if maxRadius is None:
        return None
    maxVisibleDistance += maxRadius
    return maxVisibleDistance


def feasibleRHPolygon(field, offsetL, offsetR,
                      tField, tOffsetL, tOffsetR,
                      lowerBound, upperBound, maxDist):
    """Find where objects aligned to the given fields can satisfy the given RH bounds."""
    if (offsetR - offsetL >= math.tau
            or tOffsetR - tOffsetL >= math.tau
            or upperBound - lowerBound >= math.tau):
        return None
    polygons = []
    expanded = [(poly.buffer(maxDist), heading) for poly, heading in tField.cells]
    for baseCell, baseHeading in field.cells:  # TODO skip cells not contained in base region?
        for expandedTargetCell, targetHeading in expanded:
            lower, upper = relativeHeadingRange(baseHeading, offsetL, offsetR,
                                                targetHeading, tOffsetL, tOffsetR)
            if upper >= lowerBound and lower <= upperBound:  # RH intervals overlap
                intersection = baseCell & expandedTargetCell
                if not intersection.is_empty:
                    assert isinstance(intersection, shapely.geometry.Polygon), intersection
                    polygons.append(intersection)
    return polygonUnion(polygons)


def relativeHeadingRange(baseHeading, offsetL, offsetR,
                         targetHeading, tOffsetL, tOffsetR):
    """Lower/upper bound the possible RH between two headings with bounded disturbances."""
    if baseHeading is None or targetHeading is None:  # heading may not be constant within cell
        return -math.pi, math.pi
    lower = normalize_angle(baseHeading + offsetL)
    upper = normalize_angle(baseHeading + offsetR)
    points = [lower, upper]
    if upper < lower:
        points.extend((math.pi, -math.pi))
    tLower = normalize_angle(targetHeading + tOffsetL)
    tUpper = normalize_angle(targetHeading + tOffsetR)
    tPoints = [tLower, tUpper]
    if tUpper < tLower:
        tPoints.extend((math.pi, -math.pi))
    rhs = [tp - p for tp in tPoints for p in points]  # TODO improve
    return min(rhs), max(rhs)
