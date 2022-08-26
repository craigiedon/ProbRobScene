import numpy as np
from pyrep import PyRep
from pyrep.objects.object import Object

from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs

def set_camera(cameraPos,targetPos,offset):
    """Places set camera and by the target + offset"""
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    defaultCamera = Object.get_object('DefaultCamera')
    defaultCamera.set_position(cameraPos)
    
    cameraPos = np.array(cameraPos)
    targetPos = np.array(targetPos)
    offset = np.array(offset)
    
    zaxis = normalize(targetPos - cameraPos)    
    xaxis = normalize(np.cross(offset, zaxis))
    yaxis = np.cross(zaxis, xaxis)
    
    # Arrange transformation matrix according to Coppelia's documentation
    # https://www.coppeliarobotics.com/helpFiles/en/positionOrientationTransformation.htm
    camToWorld = np.column_stack((xaxis, yaxis, zaxis, cameraPos))
    camToWorld = np.row_stack((camToWorld, np.array([0,0,0,1])))
    
    defaultCamera.set_matrix(camToWorld)


def take_screenshot(scene, 
                cameraPos=None,
                targetPos=[0,0,0],
                offset=[0,0,1]):
    """take a screenshot of a scene 
    :scene: a scene generated from scenario
    :cameraPos: camera position (if None, default is use)
    :targetPos: target position (if None, default is [0,0,0])
    :offset: offset between camera and target (if None, default is [0,0,1])
    """
    pr = PyRep()
    pr.launch("scenes/emptyVortex.ttt", headless=True)
    cop_from_prs(pr, scene)
    
    # Lua script to save a screenshot
    pr.import_model("models/screenshot.ttm")
    
    if cameraPos:
        set_camera(cameraPos, targetPos, offset)

    # save as coppelliaSim_screenshot_<timestamp>.png  
    pr.script_call("handleStuff@screenshotSensor", 6) 
    pr.shutdown()
