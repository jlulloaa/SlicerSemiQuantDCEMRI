import logging
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
    parameterPack,
)

from slicer import vtkMRMLMarkupsROINode, vtkMRMLLabelMapVolumeNode, vtkMRMLColorTableNode
from slicer import vtkMRMLSequenceNode, vtkMRMLSegmentationNode, vtkMRMLTableNode#, vtkMRMLProceduralColorNode

import numpy as np
from scipy import signal
import pydicom as pydcm

import time

#
# quantification
#


class quantification(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Semi-Quantitative DCE-MRI")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Quantification")]
        self.parent.dependencies = [] # ["SequenceRegistration"]  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Jose L. Ulloa (ISANDEX LTD.), Muhammad Ayyaz Qadir (Monash University, Austin Health)"] 
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
Slicer Extension to derive semi-quantitative parameters from DCE-MRI datasets. 
For up-to-date user guide, go to the <a href="https://github.com/jlulloaa/SlicerSemiQuantDCEMRI"> official GitHub repository </a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jose L. Ulloa and Muhammad Qadir.
It is derived from the extension <a href="https://github.com/rnadkarni2/SlicerBreast_DCEMRI_FTV"> Slicer DCEMRI FTV </a>. 
This work was (partially) funded by… (grant Name and Number).
""")


@parameterPack
class relevantDCEindices:
    
  preContrast: Annotated[float, WithinRange(0.0, 1000.0)] = 0.0
  earlyPostContrast: Annotated[float, WithinRange(0.0, 1000.0)] = 0.0
  latePostContrast: Annotated[float, WithinRange(0.0, 1000.0)] = 0.0

  
  def setDefault(self, maxIndex=0):

      self.preContrast = 0
      self.earlyPostContrast = 3
      self.latePostContrast = maxIndex-1
      

@parameterPack
class timeLabelDCEindices:
    
    preContrastTimeLabel: str=""
    earlyPostContrastTimeLabel: str=""
    latePostContrastTimeLabel: str=""


@parameterPack
class volumeSubtractionIndices:
    
  minuend: Annotated[int, WithinRange(0, 10)] = 1
  subtrahend: Annotated[int, WithinRange(0, 10)] = 0


#
# quantificationParameterNode
#
    

@parameterNodeWrapper
class quantificationParameterNode:
    """
    The parameters needed by module.

    input4DVolume - The input sequence to process.
    inputMaskVolume - The volume containing the segmentation masks.
    outputSequenceMaps - The output Parametric Maps
    outputLabelMap - The output in the form of coloured labels

    backgroundThreshold - The value at which to threshold the signal intensity images
    peakEnhancementThreshold - The value at which to threshold the Percentage Enhancement (PE) map
    signalEnhancementRatioThreshold - The value at which to threshold the Signal Enhancement Ratio (SER) map

    displaySERrange - If True (default) the output will show a coloured output based on a pre-defined SER interval range. If False, the output will use the SER threshold value to divide the output
    markupROIVisibilityToggle - If True the ROI box markup will be shown in the display
    segmentMaskVisibilityToggle - If True the segmentation mask will be shown in the dsplay
        
    indicesDCE - The relevant indices defining the Pre-Contrast, Early and Later Post-Contrast images from the input Sequence
    subtractIndices - The indices of the images in the input sequence that will be subtracted and displayed

    """

    input4DVolume: vtkMRMLSequenceNode
    inputMaskVolume: vtkMRMLSegmentationNode
    outputSequenceMaps: vtkMRMLSequenceNode
    outputLabelMap: vtkMRMLLabelMapVolumeNode

    # # JU - Widgets not yet supported by the Parameters Node Wrapper Infrastructure
    # # Check this for any update: https://github.com/Slicer/Slicer/issues/7308
    # segmentSelectorWidgetParam: qMRMLSegmentSelectorWidget
    # segmentEditorWidgetParam: qMRMLSegmentEditorWidget
    # tableTICNode: vtkMRMLTableNode
    # plotTICNode: vtkMRMLPlotSeriesNode
    # plotChartTICNode: vtkMRMLPlotChartNode

    # JU - Controllers for the analysis and display
    indicesDCE: relevantDCEindices
    timingsDCE: timeLabelDCEindices
    
    peakEnhancementThreshold: Annotated[float, WithinRange(0.0, 250.0)] = 70.0
    backgroundThreshold: Annotated[float, WithinRange(0.0, 100.0)] = 60.0
    signalEnhancementRatioThreshold: Annotated[float, WithinRange(0.0, 3.25)] = 1.4

    # # JU - Checkbox to propagate the timings for any other sequence that hasn't DICOM metadata (assumes a single patient in the scene)
    propagateTiming: bool = True

    # # JU - Simple Checkbox to control SER analysis:
    displaySERrange: bool = True

    # JU - controllers to display volume subtraction
    subtractIndices: volumeSubtractionIndices
    
    # # JU - Simple Checkboxes to control layers visibility:
    markupROIVisibilityToggle:  bool = True
    segmentMaskVisibilityToggle:  bool = True
    
#
# quantificationWidget
#


class quantificationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """


    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        # Setting up the display
        # Customise the layout before starting (https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#customize-view-layout)
        # To get more help, check the code: https://github.com/Slicer/Slicer/blob/main/Libs/MRML/Logic/vtkMRMLLayoutLogic.cxx
        customLayout = """
        <layout type="vertical" split="true" >
        <item splitSize="500">
            <layout type="vertical">
            <item>
                <layout type="horizontal">
                    <item>
                        <view class="vtkMRMLSliceNode" singletontag="Red">
                        <property name="orientation" action="default">Axial</property>
                        <property name="viewlabel" action="default">R</property>
                        <property name="viewcolor" action="default">#F34A33</property>
                        </view>
                    </item>
                    <item>
                        <view class="vtkMRMLSliceNode" singletontag="Yellow">
                        <property name="orientation" action="default">Sagittal</property>
                        <property name="viewlabel" action="default">Y</property>
                        <property name="viewcolor" action="default">#EDD54C</property>
                        </view>
                    </item>
                </layout>
            </item>
            </layout>
        </item>
        <item splitSize="300">
            <layout type="vertical">
            <item>
                <layout type="horizontal">
                    <item>
                        <view class="vtkMRMLPlotViewNode" singletontag="PlotView1">
                        <property name="viewlabel" action="default">P</property>
                        </view>
                    </item>
                    <item>
                        <view class="vtkMRMLTableViewNode" singletontag="TableView1">
                        <property name="viewlabel" action="default">T</property>
                        </view>
                    </item>
                </layout>
            </item>
            </layout>
        </item>
        </layout>
        """

        # Built-in layout IDs are all below 100, so we can choose any large random number
        # for your custom layout ID.
        self.customLayoutId=990

        # The ID for the 3D-render-view-only is 4:
        self.volumeRenderOnlyLayout = 4
        
        # JU - Setting up a layout manager object
        self.layoutManager = slicer.app.layoutManager()

        # Add the custom layout:
        self.layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(self.customLayoutId, customLayout)

        # JU - Switch to a layout that contains a plot view to create a plot widget.
        # 38 is the layout called "Four-up Quantitative" in the layout dropdown list 
        # (to check which number is currently set, use: slicer.app.layoutManager().layout in slicer's console)
        # self.layoutManager.setLayout(38)
        # Switch to the new custom layout
        self.layoutManager.setLayout(self.customLayoutId)

        # Ensure the markers are visible in all the views:
        viewNodes = slicer.util.getNodesByClass("vtkMRMLAbstractViewNode")

        for viewNode in viewNodes:
            viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeAxes)

        # Display the slice intersections:
        sliceDisplayNodes = slicer.util.getNodesByClass("vtkMRMLSliceDisplayNode")

        for sliceDisplayNode in sliceDisplayNodes:
            sliceDisplayNode.SetIntersectingSlicesVisibility(1)

        # JU - End setting up the layout and display
        
        # JU - To ensure the columns name are consisten between TICTable and TICplot, I define them here:
        self.TICTableRowNames = ["Timepoint [min]", "PE (%)", "Linear Fit"]
        self.SummaryTableRowNames = ["Parameter", "Value", "Units"]
        self.SERTableRowNames = ["Ranges", "Volume (cm3)", "Distribution (%)"]

        # JU - Auxiliar nodes and variables
        self.currentVolume = None
        self.subtractVolume = None
        self.roiNode = None
        self.segmentID = None
        self.colourTableNode = None
        self.timeFrames = None
        self.serMapInterval = None
        # Anything above SER_UPPER_THRESHOLD will be considered NON-SER (together with negative values). 
        # This is to be consistent with FTVDCEMRI and Aegis, where everything above 3.0 is not considered.
        # By doing this, I also simplify the labelling and SER map derivation, as the upper threshold is already cut with this value
        self.SER_UPPER_THRESHOLD = 3.0
        # When selecting a single SER Threshold, the intervals are defined by:
        # 0 < SER ≤ SER_Threshold
        # SER_Threshold < SER ≤  SER_Threshold * (1 + SER_DELTA_FACTOR)
        # self.SER_DELTA_FACTOR = 0.1 
        self.SER_DELTA_FACTOR = 4/9 # to be consistent with the default 0.0 - 0.9 - 1.3
        
        self.useSERmap = True
        
        # Clear up the roi boxes
    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/quantification.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
                
        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = quantificationLogic()
        
        # # JU - Create segment editor to get access to effects
        # # JU - To show segment editor widget (useful for debugging): segmentEditorWidget.show()
        self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(self.segmentEditorNode)
        self.ui.segmentEditorWidget.enabled = False
        self.ui.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
             
        # JU - Output table selector - 
        self.outputTableSelector = slicer.qMRMLNodeComboBox()
        self.outputTableSelector.noneDisplay = _("Create new table")
        self.outputTableSelector.setMRMLScene(slicer.mrmlScene)
        self.outputTableSelector.nodeTypes = ["vtkMRMLTableNode"]
        self.outputTableSelector.enabled = False
        self.outputTableSelector.addEnabled = False
        self.outputTableSelector.selectNodeUponCreation = True
        self.outputTableSelector.renameEnabled = False
        self.outputTableSelector.removeEnabled = False
        self.outputTableSelector.noneEnabled = False
        self.outputTableSelector.setToolTip(_("Select a Table"))
        self.outputTableSelector.setCurrentNode(None)
        # use insertRow to append the new element into a specific row (e.g. FormLayout.insertRow(position, Text, Table))
        # or a more general way by using addWidget:
        self.ui.outputGridLayout.addWidget(self.outputTableSelector, 4, 1, 1)
        
        # Connections - Here there are the connections that aren't made automatically by the Parameter node wrapper
        # JU - Because segmentEditor and Selector widgets are not yet supported by the parameters node wrapper, 
        #     their connections to the scene have be done manually:
        self.ui.segmentEditorWidget.segmentationNodeChanged.connect(self.onSegmentChangeSegmentEditorNode)
        self.ui.inputMaskSelector.currentNodeChanged.connect(self.onNodeChangeInputMaskSelectorNode)
        # JU - This connection manages the visibility of the segment mask
        self.ui.segmentSelectorWidget.currentSegmentChanged.connect(self.updateSelectedSegmentMask)

        # JU - This connection manages the visibility of the table
        self.outputTableSelector.currentNodeChanged.connect(self.onSelectDisplayTable)
        
        # JU - Radio buttons are not avilable yet in the parameter node infraestructure
        self.ui.defaultLayoutViewRadioButton.connect("clicked()", self.checkDefaultVieweLayout)
        self.ui.renderingLayoutViewRadioButton.connect("clicked()", self.checkDefaultVieweLayout)

        # JU 09/07/2025 - Add selector to calculate SER maps or Relative Enhancement (CAD-like)
        self.ui.SERmapsLayoutViewRadioButton.connect("clicked()", self.checkSERmapsCalcVieweLayout)
        self.ui.CADmapsLayoutViewRadioButton.connect("clicked()", self.checkSERmapsCalcVieweLayout)
        
        # JU - Because I want to display/set the current view to whatever slider is moved, I thinks a connector is required:
        self.ui.indexSliderPreContrast.connect("valueChanged(double)", self.setCurrentVolumeFromIndex)
        self.ui.indexSliderEarlyPostContrast.connect("valueChanged(double)", self.setCurrentVolumeFromIndex)
        self.ui.indexSliderLatePostContrast.connect("valueChanged(double)", self.setCurrentVolumeFromIndex)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons have to be connected manually, because they are controlled manually by the user:
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.sequenceRegistrationButton.connect("clicked(bool)", self.goToSequenceRegistration)
        self.ui.displaySubtractionButton.connect("clicked(bool)", self.onDisplaySubtractionVolumes)
        self.ui.resetSegmentListButton.connect("clicked(bool)", self.onResetSegmentList)
        
        # Button to add OMIT regions:
        self.ui.addOmitRegionButton.connect("clicked(bool)", self.onAddOmitRegion)
        # (Re)Set the OMIT region counter:
        self.omit_counter = 0
        # JU - Everytime the module is reloaded, just delete the omit regions (would this be the correct behaviour??)
        self.cleanUpRoiBoxNodes()
        self.omitRoiList = []

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()


    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        
        self.removeObservers()


    def enter(self) -> None:
        """Called each time the user opens this module."""
        
        # Make sure parameter node exists and observed
        self.initializeParameterNode()


    def exit(self) -> None:
        """Called each time the user opens a different module."""
        
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)


    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)


    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()


    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.input4DVolume:
            firstInputSequenceNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSequenceNode")

            if firstInputSequenceNode:
                self._parameterNode.input4DVolume = firstInputSequenceNode

        # # JU - for the SER threshold slider, set the maximum to the UPPER_THRESHOLD:
        self.ui.signalEnhancementRatioThreshold.maximum = self.SER_UPPER_THRESHOLD / (1.0 + self.SER_DELTA_FACTOR)

        # JU - Initialise SERsegmentsLabels for SER values. It has to happens after the _parameterNode is created
        self.setSERColourMapDict(update=True)
        # JU 08/07/2025 - Initialises CADsegmentsLabels to allow CAD-like maps (Washout/Persistent/Plateau)
        self.setCADColourMapDict()

        # JU 12/06/2024 - The following should happen only if input4D volume exist
        # Initialise the output sequence that'll store the output maps, but only if the input sequence has been defined:

        if self._parameterNode.input4DVolume:
            self._parameterNode.indicesDCE.setDefault(self._parameterNode.input4DVolume.GetNumberOfDataNodes())
            self.setupBoxROI()

            if not self._parameterNode.outputSequenceMaps:
                self._parameterNode.outputSequenceMaps = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "OutputSequenceNode")
                # Set the index to be maps names:
                self._parameterNode.outputSequenceMaps.SetIndexName("Maps")
                self._parameterNode.outputSequenceMaps.SetIndexType(1)  # 0: Numeric; 1: Text
                self._parameterNode.outputSequenceMaps.SetIndexUnit("")
                self.outputSeqBrowser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "OutputSequenceBrowserNode")
                self.outputSeqBrowser.SetAndObserveMasterSequenceNodeID(self._parameterNode.outputSequenceMaps.GetID())

            # # Select default input nodes if nothing is selected yet to save a few clicks for the user
            if not self._parameterNode.inputMaskVolume:
                firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
                
                if firstVolumeNode:
                    self._parameterNode.inputMaskVolume = firstVolumeNode
                elif self._parameterNode.input4DVolume:
                    self._parameterNode.inputMaskVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation Mask")

            # Now that the segmentation mask has been created, add a default segmentation to initialise it:
            if self._parameterNode.inputMaskVolume:
                # Get the number of segmentations attached to the segmentation node:
                segmentations = self._parameterNode.inputMaskVolume.GetSegmentation()
                
                if segmentations.GetNumberOfSegments() < 1:
                    # Create a new segmentation and attach it to the inputMaskVolume node:
                    segmentations.AddEmptySegment()
                    
                self.ui.segmentEditorWidget.setSegmentationNode(self._parameterNode.inputMaskVolume)
                self.ui.segmentEditorWidget.setSourceVolumeNode(self.currentVolume)
                
                self.segmentID = self.ui.segmentSelectorWidget.currentSegmentID()

                # Open the segment editor
                self.ui.segmentEditorCollapsibleButton.collapsed=False
                
            # # Define a default output Label Map:
            if not self._parameterNode.outputLabelMap:
                firstOutputLabelMap = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
                
                if firstOutputLabelMap:
                    self._parameterNode.outputLabelMap = firstOutputLabelMap
                else:
                    # There are no output label map available, so create one by default:
                    self._parameterNode.outputLabelMap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "SER Label Map")

            # Create a default display node, so I can associate the colour table:
            self._parameterNode.outputLabelMap.CreateDefaultDisplayNodes()

            ser_labels_colour_table = slicer.mrmlScene.GetNodesByName("SER_labels")
            if (self.colourTableNode is None):
                # colourTableNode does not exist, let see whether the SER_labels colour table already exist
                if ser_labels_colour_table.GetNumberOfItems() > 0:
                    # Then assign the existing table:
                    self.colourTableNode = ser_labels_colour_table.GetItemAsObject(0)
                else:
                    # JU 08/07/2025 - Let's try procedural Colour node (https://discourse.slicer.org/t/difference-and-use-cases-for-vtkmrmlproceduralcolornode-and-vtkmrmlcolortablenode/24602/1 )
                    # self.colourTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLProceduralColorNode", "SER_labels")
                    self.colourTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", "SER_labels")

            self.colourTableNode.SetTypeToUser()

            # make the color table selectable in the GUI outside Colors module
            self.colourTableNode.HideFromEditorsOff()
            self.setupColourTable()

            # # Associate the colour table with the label map --> Pay attention to the use cases, because there is an error at some point (not yet clear when though)
            self._parameterNode.outputLabelMap.GetDisplayNode().SetAndObserveColorNodeID(self.colourTableNode.GetID())

            
            # Select default plot and tables nodes, to avoid creating new ones:
            if not self.outputTableSelector.currentNode():
                self.TICTableNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLTableNode")
                self.SummaryTableNode = slicer.mrmlScene.GetNthNodeByClass(1, "vtkMRMLTableNode")
                self.SERDistributionTableNode = slicer.mrmlScene.GetNthNodeByClass(2, "vtkMRMLTableNode")

                if not self.TICTableNode:
                    self.TICTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "TIC Table")

                # TODO: Check how to assign multiple tables to selector
                if not self.SummaryTableNode:
                    self.SummaryTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Summary Table")

                if not self.SERDistributionTableNode:
                    self.SERDistributionTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "SER Table")

                self.outputTableSelector.setCurrentNode(self.SERDistributionTableNode)

            numberOfPlotSeriesNode = slicer.mrmlScene.GetNodesByClass("vtkMRMLPlotSeriesNode").GetNumberOfItems()

            if numberOfPlotSeriesNode == 0:
                firstPlotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "TIC plot")
                secondPlotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Linear Fit")
            elif numberOfPlotSeriesNode == 1:
                firstPlotSeriesNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLPlotSeriesNode")
                secondPlotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Linear Fit")
            else:
                firstPlotSeriesNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLPlotSeriesNode")
                secondPlotSeriesNode = slicer.mrmlScene.GetNthNodeByClass(1, "vtkMRMLPlotSeriesNode")
                
            self.plotSeriesNode = firstPlotSeriesNode
            self.plotCurveFitNode = secondPlotSeriesNode
            self.plotSeriesNode.SetAndObserveTableNodeID(self.TICTableNode.GetID())
            self.plotCurveFitNode.SetAndObserveTableNodeID(self.TICTableNode.GetID())

            firstPlotChartNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLPlotChartNode")

            if not firstPlotChartNode:
                firstPlotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "TIC chart")

            self.plotChartNode = firstPlotChartNode

            # Ensure there is no previous charts in the node (to avoid multiple legends appearing when re-loading):
            self.plotChartNode.RemoveAllPlotSeriesNodeIDs()
                    
            self.plotChartNode.AddAndObservePlotSeriesNodeID(self.plotSeriesNode.GetID())
            self.plotChartNode.AddAndObservePlotSeriesNodeID(self.plotCurveFitNode.GetID())
            
            # Finally, (re-)configure the plot window
            self.configurePlotSeriesNode()
     
     
    def setParameterNode(self, inputParameterNode: Optional[quantificationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            
            if self._parameterNode.input4DVolume:
                # Setup default positions for the sliders: preContrast = 0; earlyPostContrast = 1; latePostContrast = N_Nodes
                self.setMaxIndexSelector(self._parameterNode.input4DVolume.GetNumberOfDataNodes())
                self.setCurrentVolumeFromIndex(indexAsDouble=self._parameterNode.indicesDCE.earlyPostContrast)                
                self._checkCanApply()
                

    def _checkCanApply(self, caller=None, event=None) -> None:

        if self._parameterNode and self._parameterNode.input4DVolume:

            self.setMaxIndexSelector(self._parameterNode.input4DVolume.GetNumberOfDataNodes())
            self.setCurrentVolumeFromIndex()

            self.ui.relevantIndicesCollapsibleButton.enabled = True
            self.ui.relevantIndicesCollapsibleButton.collapsed = False
            self.ui.displaySubtractionCollapsibleButton.enabled = True
            self.ui.displaySubtractionCollapsibleButton.collapsed = False
            self.ui.displaySubtractionButton.enabled=True
            self.ui.sequenceRegistrationButton.enabled = True

            # JU - Enable the omit regions 
            self.ui.addOmitRegionButton.enabled = True
            # Check the visibility of the ROI
            self.toggleROIsView() # ===> Add an update to the ROI so it can reset when loading new data

            # Update the sequence times:
            self.timings = self.getAcquisitionTimings()

            # Update the display with the refreshed sequence times
            self.setTimeValueOnSlider()

            # If not yet defined, initialise the colour map dictionary
            self.setSERColourMapDict()

            if self._parameterNode.inputMaskVolume:
                self.ui.parametersCollapsibleButton.enabled=True
                self.ui.parametersCollapsibleButton.collapsed=False
                self.ui.peakEnhancementThreshold.enabled = True
                self.ui.signalEnhancementRatioThreshold.enabled = not self._parameterNode.displaySERrange
                self.ui.backgroundThreshold.enabled = True
                self.ui.segmentEditorWidget.enabled = True
                self.ui.outputsCollapsibleButton.enabled = True
                self.ui.outputsCollapsibleButton.collapsed = False
                self.outputTableSelector.enabled = True
                self.ui.resetSegmentListButton.enabled = True
                if self._parameterNode.displaySERrange:
                    self.ui.SERpredefinedRangeToggle.text=f"Pre-defined SER range: [{', '.join([f'{xi:.2f}' for xi in self.serMapInterval])}]"
                else:
                    self.ui.SERpredefinedRangeToggle.text=f"Single SER threshold, range: [{', '.join([f'{xi:.2f}' for xi in self.serMapInterval])}]"
                self.ui.applyButton.toolTip = _("Compute FTV parameters")
                self.ui.applyButton.enabled = True
               
        else:
            self.ui.relevantIndicesCollapsibleButton.enabled = False
            self.ui.relevantIndicesCollapsibleButton.collapsed = False
            self.ui.displaySubtractionCollapsibleButton.enabled = False
            self.ui.displaySubtractionCollapsibleButton.collapsed = True
            self.ui.displaySubtractionButton.enabled=False
            self.ui.segmentEditorCollapsibleButton.collapsed=True
            self.ui.parametersCollapsibleButton.enabled=False
            self.ui.parametersCollapsibleButton.collapsed=True
            self.ui.applyButton.toolTip = _("Select input sequence volume")
            self.ui.applyButton.enabled = False
       
            
    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):

            # Close the segment editor
            self.ui.segmentEditorCollapsibleButton.collapsed=True
            
            # Update timings including bolus time (if available from the DICOM metadata)
            self.timings = self.getAcquisitionTimings(getBolus=True)

            # If there are previous SER maps, removes them from the segment list:
            # Ensure to delete all elements in the list (to allow switching between maps)
            self.logic.resetSegmentList(self._parameterNode.inputMaskVolume, items_to_remove=self.SERsegmentsLabels['legend'] + self.CADsegmentsLabels['legend'])
            
            if self.useSERmap:
                # self.logic.resetSegmentList(self._parameterNode.inputMaskVolume, items_to_remove=self.SERsegmentsLabels['legend'])
                # Update SERsegmentsLabels with the value in the SER theshold slider:
                self.setSERColourMapDict(update=True, serUpperThreshold=self.SER_UPPER_THRESHOLD)
            else:
                # self.logic.resetSegmentList(self._parameterNode.inputMaskVolume, items_to_remove=self.CADsegmentsLabels['legend'])          
                self.setCADColourMapDict()

            self.setupColourTable()

            # Update the list of omit regions to skip those removed manually:
            self.updateOmitRegionList()
            
            # If list of omit regions is not empty, it is safe to reset the segmentation mask to redo the calculations
            if self.omitRoiList:
                self.onResetSegmentList()

            # Compute output
            self.logic.process(self._parameterNode.input4DVolume, 
                               self._parameterNode.inputMaskVolume, 
                               self._parameterNode.outputSequenceMaps, 
                               self._parameterNode.outputLabelMap,
                               self.roiNode,
                               self.SERsegmentsLabels,
                               self.CADsegmentsLabels,
                               self.colourTableNode, # JU 09/07/2025 - Pass the colourTableNode to enable update it with the analysis results
                               self.omitRoiList,
                               {'TICTable': [self.TICTableNode, self.TICTableRowNames],
                                'SummaryTable': [self.SummaryTableNode, self.SummaryTableRowNames],
                                'SERSummaryTable': [self.SERDistributionTableNode, self.SERTableRowNames]},
                               int(self._parameterNode.indicesDCE.preContrast),
                               int(self._parameterNode.indicesDCE.earlyPostContrast),
                               int(self._parameterNode.indicesDCE.latePostContrast),
                               self.timings,
                               self._parameterNode.peakEnhancementThreshold,
                               self._parameterNode.backgroundThreshold,
                               self.segmentID,
                               self.SER_UPPER_THRESHOLD,
                               self.useSERmap
                               )
            self.update_plot_window()
            
            try:
                # Set the X-range of the graph to be the even number closest to 125% of the maximum
                xRange = np.array(self.TICTableNode.GetTable().GetColumn(0).GetFiniteRange())
                xMax = np.floor(xRange[1] * 1.25)
                xLims = [xRange[0],  xMax + ( xMax % 2 )]
                self.plotChartNode.XAxisRangeAutoOff()
                self.plotChartNode.SetXAxisRange(xLims)
            except:
                logging.error('Cannot set the limits in the X-axis')


    # JU - user-defined connnector functions
    def goToSequenceRegistration(self) -> None:
        
        # Warn that it will move away from the module
        warnText = "If the extension 'SEQUENCE REGISTRATION' is installed, it will leave this module.\n\nTo come back here, go to:\nModules -> Quantification -> Parameteric DCE-MRI"
        ok = slicer.util.confirmOkCancelDisplay(warnText, windowTitle="WARNING - Leaving this module")
        
        if ok:
            slicer.util.selectModule("SequenceRegistration")


    def checkDefaultVieweLayout(self) -> None:
        
        if self.ui.defaultLayoutViewRadioButton.checked:
            self.layoutManager.setLayout(self.customLayoutId)
        else:
            self.layoutManager.setLayout(self.volumeRenderOnlyLayout)
            
    # def checkVolumeRenderingVieweLayout(self) -> None:

    #     if self._parameterNode.renderingLayoutViewToggle:
    #         self.layoutManager.setLayout(self.volumeRenderOnlyLayout)
    #     else:
    #         self.layoutManager.setLayout(self.customLayoutId)

    #     self._parameterNode.defaultLayoutViewToggle = not self._parameterNode.renderingLayoutViewToggle
       
    def checkSERmapsCalcVieweLayout(self) -> None:
        
        if self.ui.SERmapsLayoutViewRadioButton.checked:
            self.ui.SERpredefinedRangeToggle.enabled=True
            self.useSERmap = True
        else:
            self.useSERmap = False
            self.ui.SERpredefinedRangeToggle.enabled=False

            
    def toggleROIsView(self) -> None:
        
        if self.roiNode:
            self.roiNode.SetDisplayVisibility(self._parameterNode.markupROIVisibilityToggle)
        
        self.updateOmitRegionList()
        if self.omitRoiList:
            for omitRegionNode in self.omitRoiList:
                omitRegionNode.SetDisplayVisibility(self._parameterNode.markupROIVisibilityToggle)

        if self.segmentID:
            maskDisplayNode = self._parameterNode.inputMaskVolume.GetDisplayNode()
            maskDisplayNode.SetSegmentVisibility(self.segmentID, self._parameterNode.segmentMaskVisibilityToggle) 
        
    
    def onSegmentChangeSegmentEditorNode(self) -> None:
        
        self.ui.inputMaskSelector.setCurrentNode(self.ui.segmentEditorWidget.segmentationNode())
        
    
    def onNodeChangeInputMaskSelectorNode(self) -> None:
        
        self.ui.segmentEditorWidget.setSegmentationNode(self.ui.inputMaskSelector.currentNode())
    
    
    def onSequenceChangeInputSelectorNode(self) -> None:

        if self._parameterNode is not None:
            self._parameterNode.input4DVolume = self.ui.inputSelector.currentNode()
            self.onInputSelect()
            self.timings = self.getAcquisitionTimings()

        
    def onInputSelect(self):
        
        if not self._parameterNode.input4DVolume: #self.ui.inputSelector.currentNode():
            numberOfDataNodes = 0
            
        else:
            numberOfDataNodes = self._parameterNode.input4DVolume.GetNumberOfDataNodes() #self.ui.inputSelector.currentNode().GetNumberOfDataNodes()

        logging.debug(f'Number of items in the sequence: {numberOfDataNodes}')
        
        self.setMaxIndexSelector(numberOfDataNodes)
        
    
    def updateSelectedSegmentMask(self) -> None:

        self.segmentID = self.ui.segmentSelectorWidget.currentSegmentID()

        # # Set visibility for the selected segmentation label:
        if self._parameterNode:
            displayNode = self._parameterNode.inputMaskVolume.GetDisplayNode()
            displayNode.SetAllSegmentsVisibility(False) # Hide all segments
            displayNode.SetSegmentVisibility(self.segmentID, True)

        # JU - TODO: set viewer to the slice where the roi can be seen (decide if using the first, middle, last or any other relevant for the study)

    
    # From MultiVolumeImporterPlugin 
    # (https://github.com/fedorov/MultiVolumeImporter/blob/c8a37eb5e4f35b78ccc9287b298457a064c9d001/MultiVolumeImporterPlugin.py#L779)
    def tm2ms(self,tm) -> None:
        
        if len(tm)<6:
            return 0

        try:
            hhmmss = tm.split('.')[0]
        except:
            hhmmss = tm

        try:
            ssfrac = float('0.'+tm.split('.')[1])
        except:
            ssfrac = 0.

        if len(hhmmss)==6: # HHMMSS
            sec = float(hhmmss[0:2])*60.*60.+float(hhmmss[2:4])*60.+float(hhmmss[4:6])
        elif len(hhmmss)==4: # HHMM
            sec = float(hhmmss[0:2])*60.*60.+float(hhmmss[2:4])*60.
        elif len(hhmmss)==2: # HH
            sec = float(hhmmss[0:2])*60.*60.
        else:
            raise OSError("Invalid DICOM time string: "+tm+" (failed to parse HHMMSS)")

        sec = sec+ssfrac

        return sec*1000.
        
    
    def getAcquisitionTimings(self, getBolus=False):
        
        if self._parameterNode.input4DVolume is not None:
            
            if self._parameterNode.input4DVolume.GetAttribute("MultiVolume.FrameLabels") is not None:
                
                # Get acquisition times from DICOM metadata (output in ms)
                self.timeFrames = np.array(self._parameterNode.input4DVolume.GetAttribute("MultiVolume.FrameLabels").split(',')).astype(float)
            else:
                if ( (self.timeFrames is None) or (not self._parameterNode.propagateTiming) ):
                    
                    nt = self._parameterNode.input4DVolume.GetNumberOfDataNodes()
                
                    # normalise the time axis to nt = 1min = 60x10e3 [ms] to be consistent with the calculations  
                    self.timeFrames = np.linspace(0, nt*60.0*1.0e3, num=nt, endpoint=True)

            logging.debug(f'Timeframe Labels: {self.timeFrames} (ms)')
            
            if getBolus:
                
                # Identify Bolus injection time
                # Get the bolus time relative to the start of the acquisition time
                # Unfortunately, still have to read all the files to derive it
                fileList = []
                acquisitionTimes = []
                bolusInjTimes = []
                
                # Contrast Bolus Start Time is recorded in different Dicom attributes, 
                # depending on the manufacturer and sequence, so far, the following are known:
                # Philips - Dyn eThrive: (0018, 1042) ContrastBolusStartTime
                contrastBolusAttribute = ['ContrastBolusStartTime']

                if self._parameterNode.input4DVolume.GetAttribute("MultiVolume.FrameFileList") is not None:
                    fileList = self._parameterNode.input4DVolume.GetAttribute("MultiVolume.FrameFileList").split(',')

                for ifile in fileList:
                    dcmMetaData = pydcm.dcmread(ifile, stop_before_pixels=True)
                    acquisitionTimes.append((dcmMetaData.AcquisitionTime))
                    for cb_attr in contrastBolusAttribute:
                        if cb_attr in dcmMetaData:
                            bolusInjTimes.append((dcmMetaData.ContrastBolusStartTime))
                            break
                        
                if acquisitionTimes:
                    acquisitionTimeStart = sorted(acquisitionTimes)[0]
                else:
                    acquisitionTimeStart = '0'

                if bolusInjTimes:
                    bolusInjTimeList = sorted(bolusInjTimes)[0]
                    bolusInjTimeRelativeToStart = self.tm2ms(bolusInjTimeList) - self.tm2ms(acquisitionTimeStart)
                else:
                    bolusInjTimeList = '0'
                    bolusInjTimeRelativeToStart = 0.0


            else:
                bolusInjTimeRelativeToStart = 0.0
                
            timings = {'timepoints': self.timeFrames, 
                       'injectionTime': bolusInjTimeRelativeToStart}
            
        else:
            timings = None
            
        return timings


    def setTimeValueOnSlider(self) -> None:
        
        if self.timeFrames is not None:
            self.ui.labelTimePreContrast.text = f'{self.timeFrames[int(self._parameterNode.indicesDCE.preContrast)]/(1000*60):.1f}[min]'
            self.ui.labelTimeEarlyPostContrast.text = f'{self.timeFrames[int(self._parameterNode.indicesDCE.earlyPostContrast)]/(1000*60):.1f}[min]'
            self.ui.labelTimeLatePostContrast.text = f'{self.timeFrames[int(self._parameterNode.indicesDCE.latePostContrast)]/(1000*60):.1f}[min]'

        
    def setupBoxROI(self, name="RefBox", omitBox=False) -> None:
        
        roiNodes = slicer.mrmlScene.GetNodesByClassByName("vtkMRMLMarkupsROINode","RefBox")
        if roiNodes.GetNumberOfItems() < 1:
            self.roiNode = None
        else:
            # Just to keep it simple, the first RefBox is the main one:
            self.roiNode = roiNodes.GetItemAsObject(0)
        
        if self._parameterNode is not None:
            # Added support to add additional boxes as OMIT regions
            # First, define the main RefBox:
            if ( self.roiNode is None) & (self.currentVolume is not None ) & ( not omitBox ):
                # setup the ROI
                self.roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "RefBox")

                # JU - Setup a Box ROI to crop the volume of interest. When running this function, the input volume must have been defined:
                # Fit the ROI to the volume on display:
                self.logic.fitBoxROImarkupToVolume(self.currentVolume, self.roiNode)

                # Set the initial dimensions to be a fraction of the volume size
                halfSize = tuple([dim/4.0 for dim in self.roiNode.GetSize()])
                self.roiNode.SetSize(halfSize)
                new_roi_bounds = [0]*6
                self.roiNode.GetBounds(new_roi_bounds)
                logging.debug(f'ROI Box Size: {self.roiNode.GetSize()}')
            elif omitBox:
                # If an omit region is created, make dissapears the Segment Editor:
                self.ui.segmentEditorWidget.enabled=False
                # self.ui.segmentEditorWidget.collapsed=True
                self.ui.segmentEditorCollapsibleButton.collapsed=True
                # In order to create an omit region, the main RefBox must be defined (which is controlled by the previous if statement)
                # we use the RefBox coordinates to define the initial position and size of the new omit region:
                self.newOmitRegion = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", name)
                newOmitRegionDisplayNode = self.newOmitRegion.GetDisplayNode()
                # set Color to red by default so it differentiates from the Main RefBox:
                newOmitRegionDisplayNode.SetSelectedColor(1.0, 0.0, 0.0)
                
                # RefBox Position and size:
                RefBoxPos = [0]*6
                self.roiNode.GetBounds(RefBoxPos)
                RefBoxSize = self.roiNode.GetSize()
                RefBoxCentre = self.roiNode.GetCenter()
                logging.debug(f'RefBox position: {RefBoxCentre}')
                self.newOmitRegion.SetSize((RefBoxSize[0]/2, RefBoxSize[1]*2, RefBoxSize[2]))
                self.newOmitRegion.SetCenter((RefBoxPos[1]+RefBoxSize[0]/4, RefBoxCentre[1], RefBoxCentre[2]))
                                
                self.omitRoiList.append( self.newOmitRegion )


    # JU - Delete omit regions:
    def cleanUpRoiBoxNodes(self, ROInameRegExp='omit') -> None:
        roiBoxNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsROINode")
        for roiBoxNode in roiBoxNodes:
            if roiBoxNode.GetName().startswith(ROInameRegExp):
                slicer.mrmlScene.RemoveNode(roiBoxNode)
        
    # JU - Update Omit Region List
    def updateOmitRegionList(self):
        
        resetSegmentList = False
        for omitRegion in self.omitRoiList:
            logging.debug(f'Region Name: {omitRegion.GetName()}')
            getOmitRegionByName = slicer.mrmlScene.GetNodesByName(omitRegion.GetName())
            if getOmitRegionByName.GetNumberOfItems() < 1:
                logging.debug(f'Omit Region "{omitRegion.GetName()}" does not exist anymore')
                self.omitRoiList.remove(omitRegion)
                # After the loop, will need to reset the segment mask
                resetSegmentList = True
        
        # Update the legend text in the GUI:
        self.ui.omit_regions.text=f'{len(self.omitRoiList)} Omit regions defined'
        if resetSegmentList:
            self.onResetSegmentList()
                     
    
    # JU - callback function to add  a new OMIT region
    def onAddOmitRegion(self) -> None:
        # Update the list of omit regions
        self.updateOmitRegionList()
        omitROIsNro = len(self.omitRoiList)
        if  omitROIsNro > 4: # self.omit_counter > 4:
            ok = slicer.util.confirmYesNoDisplay(f'There are already {omitROIsNro} OMIT regions defined. Shall we continue?', windowTitle="WARNING")
        else:
            ok = True

        if not ok:
            return
        else:
            self.omit_counter += 1            
            self.setupBoxROI(name=f'omit_{self.omit_counter:02d}', omitBox=True)
            self.ui.omit_regions.text=f'{len(self.omitRoiList)} Omit regions defined'
            
    
    # JU - separate to refresh the index selectors everytime the module is loaded (not only when the input selector changes)
    def setMaxIndexSelector(self, maxIndex) -> None:
                
        for sequenceItemSelectorWidget in [self.ui.indexSliderPreContrast, self.ui.indexSliderEarlyPostContrast, self.ui.indexSliderLatePostContrast, self.ui.minuendIndexSelector, self.ui.subtrahendIndexSelector]:

            if maxIndex < 1:
                sequenceItemSelectorWidget.maximum = 0
                sequenceItemSelectorWidget.enabled = False
            else:
                sequenceItemSelectorWidget.maximum = maxIndex-1
                sequenceItemSelectorWidget.enabled = True
                    
        
        
    def setCurrentVolumeFromIndex(self, indexAsDouble=None) -> None:

        sequenceBrowserNode = self.logic.findBrowserForSequence(self._parameterNode.input4DVolume)
        logging.debug(f'Selected Sequence from browser is {sequenceBrowserNode.GetName()}')

        if indexAsDouble is not None:
            sequenceBrowserNode.SetSelectedItemNumber(int(indexAsDouble))

        # self.currentVolume = sequenceBrowserNode.GetProxyNode(self._parameterNode.input4DVolume) #self.ui.inputSelector.currentNode())
        # JU TODO: can we replace this by an observers into a scalar node?
        self.currentVolume = sequenceBrowserNode.GetProxyNode(self.ui.inputSelector.currentNode())

        # Update the sequence browser toolbar with the sequeence selected in the input selector
        slicer.modules.sequences.toolBar().setActiveBrowserNode(sequenceBrowserNode)        

        # # JU - This displays the selected volume in the viewer
        # # (https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#show-a-volume-in-slice-views)
        self.logic.updateViewer(self.currentVolume)

    
    def onDisplaySubtractionVolumes(self):#, minuendIndex=None, sustrahendIndex=None) -> None: # requires at least four inputs: sequence Volume, minuend Volume index, sustraend Volume index and display Node:

        minuendIndex = self._parameterNode.subtractIndices.minuend

        sustrahendIndex = self._parameterNode.subtractIndices.subtrahend
        
        self.subtractVolume = self.logic.subtractVolumes(self._parameterNode.input4DVolume, minuendIndex, sustrahendIndex, self.subtractVolume)
        self.logic.updateViewer(self.subtractVolume)

    def onResetSegmentList(self):
        self.logic.resetSegmentList(self._parameterNode.inputMaskVolume)
        # If the omit region list is empty, it should give the option to create a ROI manually:
        if len(self.omitRoiList) == 0:
            self.ui.segmentEditorWidget.enabled=True
        
        
    def configurePlotSeriesNode(self) -> None:
        
        # Configure Plot Series:
        self.plotSeriesNode.SetXColumnName(self.TICTableRowNames[0])
        self.plotSeriesNode.SetYColumnName(self.TICTableRowNames[1])
        self.plotSeriesNode.SetPlotType(self.plotSeriesNode.PlotTypeScatter)
        self.plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        self.plotSeriesNode.SetColor(0, 0.6, 1.0)
        
        # Configure Plot Linear Fit
        self.plotCurveFitNode.SetXColumnName(self.TICTableRowNames[0])
        self.plotCurveFitNode.SetYColumnName(self.TICTableRowNames[2])
        self.plotCurveFitNode.SetPlotType(self.plotCurveFitNode.PlotTypeScatter)
        self.plotCurveFitNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleSolid)
        self.plotCurveFitNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        self.plotCurveFitNode.SetColor(0, 0, 0)
        
        # Configure Chart node
        self.configurePlotChartNode()

    
    def configurePlotChartNode(self) -> None:
        
        # Configure Plot Chart Window:
        self.plotChartNode.SetTitle("Time Intensity Curves")
        self.plotChartNode.SetXAxisTitle(self.TICTableRowNames[0])
        self.plotChartNode.SetYAxisTitle(self.TICTableRowNames[1])
        self.plotChartNode.LegendVisibilityOn()
        self.plotChartNode.SetLegendFontSize(10)
        self.plotChartNode.SetXAxisRangeAuto(True)
        self.plotChartNode.SetYAxisRangeAuto(True)
        
        # Assign Plot Series to Chart window:
        self.plotWidget = self.layoutManager.plotWidget(0)
        self.plotViewNode = self.plotWidget.mrmlPlotViewNode()
        self.plotViewNode.SetPlotChartNodeID(self.plotChartNode.GetID())
    
    
    def onSelectDisplayTable(self):
        self.logic.displayTable(self.outputTableSelector.currentNode())
    
        
    def update_plot_window(self) -> None:
        
        # table to display by default: 
        self.logic.displayTable(self.SERDistributionTableNode)
        # Ensure the outputTableSelector is back to the table on display:
        self.outputTableSelector.setCurrentNode(self.SERDistributionTableNode)
        
        # JU - update plot name according to the selected segment name:
        segmentations = self._parameterNode.inputMaskVolume.GetSegmentation()
        self.plotSeriesNode.SetName(segmentations.GetSegment(self.segmentID).GetName())

        # (Re-)Configure Plot Chart window:
        self.plotWidget.update()
    
    
    def setupColourTable(self) -> None:

        # JU 08/08/2025: If colour table is a procedural colour node, need to replace these lines:
        if self.useSERmap:
            segmentLabels = self.SERsegmentsLabels
        else:
            segmentLabels = self.CADsegmentsLabels
        
        nLabels = len(segmentLabels['colourMap'])
        self.colourTableNode.SetNumberOfColors(nLabels)
        self.colourTableNode.SetNamesInitialised(True) # prevent automatic color name generation

        for idx, (legend, [r,g,b,a]) in enumerate(segmentLabels['colourMap'].items()):
            success = self.colourTableNode.SetColor(idx, legend, r, g, b, a)

            if success:
                logging.debug(f'(setupColourTable) {idx}) Legend: {legend} - (success: {success})')
                    
    def setCADColourMapDict(self):
        
        # Here, we use the scoring used by Jeong et al., 2025 (https://doi.org/10.1007/s11547-025-01956-6), where Washout Delta is 
        # defined as the difference in in pixel signal intensity in the delayed contrast-enhanced images compared to the initial 
        # contra-enhanced images:
        # Persistent type   is defined as an increase in signal intensity of mode than 10% ==> DELTA > 10% (BLUE)
        # Plateau type      is defined as a change in signal intensity of less than 10% ==> || DELTA || <= 10% (YELLOW)
        # Washout type      is defined as a decline in signal intensity of more than 10% ==> DELTA < -10% (RED)
        
        # CADLevelLB, CADLevelUB = [[-100.0, -10.0, 10.0], [-10.0, 10.0, 100.0]]
        CADColourMapDictionary = {
            'non CAD': [0.0, 0.0, 0.0, 0.0], # transparent for everything else
            'Persistent': [0.0, 0.0, 1.0, 1.0], # blue colour - Persistent
            'Plateau': [1.0, 1.0, 0.0, 1.0], # yellow - Plateau
            'Washout': [1.0, 0.0, 0.0, 1.0], # red colour - Washout
            }
        CADlegend = list(CADColourMapDictionary.keys())
                
        self.CADsegmentsLabels = {'CADthreshold': 10.0,
                                  'colourMap': CADColourMapDictionary,
                                  'legend': CADlegend,
                                  'levelThreshold': 10.0}

        
    def setSERColourMapDict(self, update=False, serUpperThreshold=None):

        """ In the FTV Extension, they split the intervals as follows:
        ]0, 0.9]: Blue (0, 0, 1)
        ]0.9, 1.1]: Purple (0.5, 0, 0.5)
        ]1.1, 1.3]: Green (0, 1, 0)
        ]1.3, 1.75]: Red (1, 0, 0)
        ]1.75, 3.0]: Yellow (1, 1, 0)
        >0.0 & <3.0 (i.e. No SER): White (1, 1, 1)
        
        But in our case, we want to have the option to use only one SER threshold value (see Musall et al. 2021, https://doi.org/10.1016/S0031-3203(98)00095-8)
        """
        alfa = 1.0

        if self._parameterNode.displaySERrange:
            SERmapThreshold = 0.9 # JU 07/07/2025: To be consistent with Hylton et al. 2012 and Henderson et al. 2018, the SER threshold is used to cut the pixels for the FTV calculation
            self.serMapInterval = [0.00, 0.90, 1.30]            
            serMapColours = [[0.0, 0.0, 0.0, 0.0],  # Non-SER values: black & transparent so they can be overlaid with the MIP      
                             [0.0, 0.0, 1.0, alfa], # blue
                             [1.0, 1.0, 0.0, alfa], # yellow
                             [1.0, 0.0, 0.0, alfa], # red
                            ]
        elif (self._parameterNode.signalEnhancementRatioThreshold == 0.0):
            SERmapThreshold = 0
            self.serMapInterval = [0.0]
            serMapColours = [[0.0, 0.0, 0.0, 0.0],   # Non-SER values: black & transparent so they can be overlaid with the MIP 
                            [0.0, 0.0, 1.0, alfa], # blue
                            ]
        else:
            SERthreshold = self._parameterNode.signalEnhancementRatioThreshold
            SERupperDelta = (1.0 + self.SER_DELTA_FACTOR) * SERthreshold
            SERmapThreshold = SERthreshold # JU (07/07/2025) - To be consistent with the FTV calculations, set the SER threshold to be the selected central one
            self.serMapInterval = [0.0, np.round(SERthreshold,2), np.round(SERupperDelta,2)]
            serMapColours = [[0.0, 0.0, 0.0, 0.0], # black & transparent so it can be overlaid with the MIP
                            [0.0, 0.0, 1.0, alfa], # blue  --> 0 < SER ≤ SERthresh
                            [1.0, 1.0, 0.0, alfa], # yellow --> SERthresh < SER ≤ SERthresh*(1+delta)
                            [1.0, 0.0, 0.0, alfa], # red
                            ]
        
        if serUpperThreshold is not None:
            if serUpperThreshold > max(self.serMapInterval):
                self.serMapInterval.append(serUpperThreshold)
            serMapColours.append( [1.0, 1.0, 1.0, 0.0]) # white
                
        if update:
            SERLevelLB = self.serMapInterval[:-1]
            SERLevelUB = self.serMapInterval[1:]
            legend = 'non SER'
            SERColourMapDictionary = {legend: serMapColours[0]}
            SERlegend = [legend]

            for idx, (lb, ub) in enumerate(zip(SERLevelLB, SERLevelUB)):
                legend = f'{lb:.2f} < SER ≤ {ub:.2f}' # LB ≤ SER < UB ==> LB < SER ≤ UB
                SERColourMapDictionary[legend] = serMapColours[idx+1]
                SERlegend.append(legend)
            
            self.SERsegmentsLabels = {'SERthreshold': SERmapThreshold,
                                    'colourMap': SERColourMapDictionary,
                                    'legend': SERlegend,
                                    'levelThreshold': {'LB': SERLevelLB, 
                                                        'UB': SERLevelUB}}

#
# quantificationLogic
#


class quantificationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        
        # Constants:
        self.EPSILON = 1.0e-6
        # self.UPPER_ENH_THRESHOLD = 5.0e6 # This is no longer used (why?)
        self.PIXEL_CONNECTIVITY = 4
        self.FG_OPACITY = 0.5
        self.LB_OPACITY = 0.0
        
        
    def getParameterNode(self):
        return quantificationParameterNode(super().getParameterNode())


    # JU - User-defined functions
    def findBrowserForSequence(self, sequenceNode):

        # JU - debug
        # TODO: Check error when loading data before invoking the module for the first time:
        # [VTK] vtkMRMLSequenceBrowserNode::IsSynchronizedSequenceNode failed: sequenceNode is invalid
        # [Qt] void qMRMLSegmentEditorWidget::setSourceVolumeNode(vtkMRMLNode *)  failed: need to set segment editor and segmentation nodes first
        browserNodes = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")

        for browserNode in browserNodes:
            if browserNode.IsSynchronizedSequenceNode(sequenceNode, True):
                return browserNode

        return None
    
           
    def getSegmentList(self, maskVolumeNode):

        nsegments = maskVolumeNode.GetSegmentation().GetNumberOfSegments()
        segmentList = [maskVolumeNode.GetSegmentation().GetNthSegment(idx) for idx in range(nsegments)]

        return segmentList


    def displayTable(self, currentTable):
        slicer.app.applicationLogic().GetSelectionNode().SetActiveTableID(currentTable.GetID())
        currentTable.SetUseColumnTitleAsColumnHeader(True)  # Make column titles visible (instead of column names)
        slicer.app.applicationLogic().PropagateTableSelection()
    
               
    def updateViewer(self, backgroundVolume, foregroundVolume=None, labelVolume=None, labelOpacity=None, legend_update=None):
        # TODO: Add the option to update the legend
        if foregroundVolume is not None:
            foregroundOpacity=self.FG_OPACITY
            cornerLabel = foregroundVolume.GetName()
        else:
            foregroundOpacity = None
            cornerLabel = backgroundVolume.GetName()
        
        if labelVolume is not None:
            # TODO: Fix the boundary case when SER threshold = 0.0. In the legend appears (none)
            if legend_update is not None:
                # Reset the names of the current table:
                # legend_update['ColourNode'].ClearNames()
                nLabels = len(legend_update['dict_legend'])
                legend_update['ColourNode'].SetNumberOfColors(nLabels)
                for idx, (legend, [r,g,b,a]) in enumerate(legend_update['dict_legend'].items()):
                    if legend.startswith('non'):
                        success = legend_update['ColourNode'].SetColor(idx, '', r, g, b, a) #Force to be transparent
                    else:
                        success = legend_update['ColourNode'].SetColor(idx, legend, r, g, b, a)
                    if success:
                        # print(f'(setupColourTable) {idx}) Legend: {legend} - (success: {success})')
                        logging.debug(f'(setupColourTable) {idx}) Legend: {legend} - (success: {success})')
                                
            colorLegendDisplayNode = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(labelVolume)
            colorLegendDisplayNode.ScalarVisibilityOn()
            colorLegendDisplayNode.GetLabelTextProperty().SetFontFamilyToArial()
            colorLegendDisplayNode.GetLabelTextProperty().SetFontSize(10)
            colorLegendDisplayNode.SetTitleText(legend_update['Title'])
            colorLegendDisplayNode.SetVisibility(True)

        
        for channels in ["Red", "Yellow"]:
            view = slicer.app.layoutManager().sliceWidget(channels).sliceView()
            view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft, cornerLabel)
                    
        slicer.util.setSliceViewerLayers(background=backgroundVolume, 
                                         foreground=foregroundVolume,
                                         foregroundOpacity=foregroundOpacity,
                                         label=labelVolume, 
                                         labelOpacity=labelOpacity)
    
  
    def showVolumeRenderingMIP(self, volumeNode, useSliceViewColors=True):
        """
        Source code from: https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/volumes.html#show-volume-rendering-using-maximum-intensity-projection
        To get more help: https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#show-volume-rendering-automatically-when-a-volume-is-loaded
        Render volume using maximum intensity projection
        :param useSliceViewColors: use the same colors as in slice views.
        
        How to use it:
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        showVolumeRenderingMIP(volumeNode)        
        
        """
        
        # Get/create volume rendering display node
        volRenLogic = slicer.modules.volumerendering.logic()
        displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)

        if not displayNode:
            displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)

        # Choose MIP volume rendering preset
        if useSliceViewColors:
            volRenLogic.CopyDisplayToVolumeRenderingDisplayNode(displayNode)
        else:
            scalarRange = volumeNode.GetImageData().GetScalarRange()

            if scalarRange[1]-scalarRange[0] < 1500:
            # Small dynamic range, probably MRI
                displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName("MR-MIP"))
            else:
                # Larger dynamic range, probably CT
                displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName("CT-MIP"))

        # Switch views to MIP mode
        for viewNode in slicer.util.getNodesByClass("vtkMRMLViewNode"):
            viewNode.SetRaycastTechnique(slicer.vtkMRMLViewNode.MaximumIntensityProjection)

        # Show volume rendering
        displayNode.SetVisibility(True)


    # Delete some segments in the segment editor (by giving the list "items_to_remove" of segments to delete) or 
    # delete all segments in the list and create a new empty one (default behaviour)
    def resetSegmentList(self, segmentationVolumeNode, items_to_remove=None):
        
        maskSegmentations = segmentationVolumeNode.GetSegmentation()

        if items_to_remove is not None:

            for segment_iID in maskSegmentations.GetSegmentIDs():
                segment_i = maskSegmentations.GetSegment(segment_iID)

                if segment_i.GetName() in items_to_remove:
                    maskSegmentations.RemoveSegment(segment_i)
        else:
            # Remove all segments and create a new one empty
            for segment_iID in maskSegmentations.GetSegmentIDs():
                segment_i = maskSegmentations.GetSegment(segment_iID)
                maskSegmentations.RemoveSegment(segment_i)

            maskSegmentations.AddEmptySegment()


    # JU - Crop Volume from ROI Box:
    def cropVolumeFromROI(self, inputVolumeArray, referenceBoxROINode, referenceScalarVolumeNode):
        
        slicer.util.updateVolumeFromArray(referenceScalarVolumeNode, inputVolumeArray)

        cropVolumeLogic = slicer.modules.cropvolume.logic()
        cropVolumeParameterNode = slicer.vtkMRMLCropVolumeParametersNode()
        cropVolumeParameterNode.SetROINodeID(referenceBoxROINode.GetID())
        cropVolumeParameterNode.SetInputVolumeNodeID(referenceScalarVolumeNode.GetID())
        cropVolumeParameterNode.SetVoxelBased(True)
        cropVolumeLogic.SnapROIToVoxelGrid(cropVolumeParameterNode)  # rotates the ROI to match the volume axis directions
        cropVolumeLogic.Apply(cropVolumeParameterNode)
        croppedVolumeNode = slicer.mrmlScene.GetNodeByID(cropVolumeParameterNode.GetOutputVolumeNodeID())
        croppedVolume = slicer.util.arrayFromVolume(croppedVolumeNode)
        
        slicer.mrmlScene.RemoveNode(croppedVolumeNode)
        
        return croppedVolume
    
    def cropSequenceVolumeFromROI(self, inputSequenceArray, referenceBoxROINode, referenceScalarVolumeNode):
            
        nt = inputSequenceArray.shape[0]
        croppedSequenceVolume = []

        for idt in range(nt):
            croppedSequenceVolume.append(self.cropVolumeFromROI(inputSequenceArray[idt,:,:,:],referenceBoxROINode,referenceScalarVolumeNode))

        outputVolumeArray = np.stack(croppedSequenceVolume, axis=0)
        
        return outputVolumeArray

        
    # JU - Fitting functions
    def simple_linear_fit(self, time_axis, sample_points, norder = 1):

        # simple lin fit: y(t) = m*t + n ==> lin_params = [m_slope, n_coeff]
        lin_params = np.polyfit(time_axis, sample_points, norder)
        yeval = np.polyval(lin_params, time_axis)

        return lin_params, yeval


    def getVolumeDataFromSequence(self, sequenceNode):

        # Fill in the 4D array from the sequence node
        # https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#access-voxels-of-a-4d-volume-as-numpy-array
        nt = sequenceNode.GetNumberOfDataNodes()

        # Size of the numpy array is ordered as [nz, ny(row), nx(col)] TODO: verify row and col are correctly assigned!!
        volume0 = slicer.util.arrayFromVolume(sequenceNode.GetNthDataNode(0))
        [nz, ny, nx] = volume0.shape
        inputVolumeArray = np.zeros([nt, nz, ny, nx]) # JU to follow ITK convention for 4D volumes

        inputVolumeArray[0,:,:,:] = volume0

        for volumeIndex in range(1, nt):
            inputVolumeArray[volumeIndex, :, :, :] = slicer.util.arrayFromVolume(sequenceNode.GetNthDataNode(volumeIndex))
        
        return inputVolumeArray

        
    def subtractVolumes(self, inputSequenceNode, minuendIndex, subtrahendIndex, outputVolumeNode=None):
        
        # if the outputVolume does not exist, creates a new one and returns it:
        if outputVolumeNode is None:
            outputVolumeNode = slicer.modules.volumes.logic().CloneVolume(inputSequenceNode.GetNthDataNode(minuendIndex), "substractionVolumeNode")
            
        minuendVolume = slicer.util.arrayFromVolume(inputSequenceNode.GetNthDataNode(minuendIndex))
        subtrahendVolume = slicer.util.arrayFromVolume(inputSequenceNode.GetNthDataNode(subtrahendIndex))
        # Ensure each volume is float, so the dynamic range is ok to represent the data without clipping it
        subtractedVolume = np.abs(minuendVolume.astype(float) - subtrahendVolume.astype(float))
        
        slicer.util.updateVolumeFromArray(outputVolumeNode, subtractedVolume)
        outputVolumeNode.SetName(f'ABS[Volume({minuendIndex})-Volume({subtrahendIndex})]')

        return outputVolumeNode
        

    def getStatsFromMask(self, volumeMaskNode, segmentID):
        
        # To get Summary statistics use the SegmentStatistics module
        # It returns a dictionary with the statistics corresponding to the segmentID provided
        # To get the complete list of statistics use:
        # import SegmentStatistics
        # SegmentStatistics.SegmentStatisticsLogic().getParameterNode().GetParameterNames()

        import SegmentStatistics
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", volumeMaskNode.GetID())
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_origin_ras.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_diameter_mm.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_direction_ras_x.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_direction_ras_y.enabled",str(True))
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.obb_direction_ras_z.enabled",str(True))
        segStatLogic.computeStatistics()
        stats = segStatLogic.getStatistics()
        outputStats = {}

        for statPlugInName, measurementDetails in stats['MeasurementInfo'].items():
            outputStats[statPlugInName.replace('LabelmapSegmentStatisticsPlugin.','')] = measurementDetails
            outputStats[statPlugInName.replace('LabelmapSegmentStatisticsPlugin.','')]['value'] = stats[segmentID, statPlugInName]

        return outputStats


    # Get ROI coordinates:
    def getBoxROIIJKCoordinates(self, markupROINode, referenceVolumeNode, transformedVolume=False):
        
        markupROI_RAS = self.getRASmarkupROICoordinates(markupROINode)

        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        if transformedVolume:
            transformRasToVolumeRas = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, referenceVolumeNode.GetParentTransformNode(), transformRasToVolumeRas)
            
            markupROI_RAS['RASmin'] = transformRasToVolumeRas.TransformPoint(markupROI_RAS['RASmin'])
            markupROI_RAS['RASmax'] = transformRasToVolumeRas.TransformPoint(markupROI_RAS['RASmax'])
        
        markupROI_IJK_inRefVol = self.convertRAStoIJKVolumeNodeCoordinates(markupROI_RAS, referenceVolumeNode)
           
        return markupROI_IJK_inRefVol
    

    def getRASmarkupROICoordinates(self, markupBoxROINode):
        
        boundingBoxRASCoordinates = np.zeros(6)
        markupBoxROINode.GetBounds(boundingBoxRASCoordinates)
        # Get BOX corners in RAS:
        bbox_rmin, bbox_rmax, bbox_amin, bbox_amax, bbox_smin, bbox_smax = boundingBoxRASCoordinates
        outputRAS = {'RASmin': [bbox_rmin, bbox_amin, bbox_smin],
                        'RASmax': [bbox_rmax, bbox_amax, bbox_smax]}
        
        return outputRAS
    

    def convertRAStoIJKVolumeNodeCoordinates(self, RAScoordinatesDict, referenceVolumeNode):
        
        volumeSize = referenceVolumeNode.GetImageData().GetDimensions() # IJK
        bbox_ijk = np.ones((2,4))
        referenceDimensions = np.full_like(bbox_ijk, np.append(volumeSize,2), dtype=int) - 1
        
        volumeRasToIjk = vtk.vtkMatrix4x4()
        referenceVolumeNode.GetRASToIJKMatrix(volumeRasToIjk)
        volumeRasToIjk.MultiplyPoint(np.append(RAScoordinatesDict['RASmin'], 1.0), bbox_ijk[0,:])

        volumeRasToIjk.MultiplyPoint(np.append(RAScoordinatesDict['RASmax'], 1.0), bbox_ijk[1,:])
        
        # Round the elements and convert them to integer:
        bbox_ijk = bbox_ijk.round().astype(int)
    
        # In case any coordinate is outside the volume, set it to 0 or max:
        bbox_ijk[bbox_ijk < 0] = 0
        outsideIndices = ( bbox_ijk > referenceDimensions )
        bbox_ijk[outsideIndices] = referenceDimensions[outsideIndices]
        
        outputIJK = {'IJKmin': np.min(bbox_ijk[:,:-1], axis=0),
                        'IJKmax': np.max(bbox_ijk[:,:-1], axis=0)+1}
    
        return outputIJK
    
    
    def getBoxROIOriginCoordinates(self, markupBoxROINode):
        
        roiDiameter = markupBoxROINode.GetSize()
        roiOrigin_Roi = [-roiDiameter[0]/2, -roiDiameter[1]/2, -roiDiameter[2]/2, 1]
        roiToRas = markupBoxROINode.GetObjectToWorldMatrix()
        roiOrigin_Ras = roiToRas.MultiplyPoint(roiOrigin_Roi)

        return roiToRas, roiOrigin_Ras
    
    
    def translateVolumeToROIBox(self, volumeNodeToTranslate, markupBoxROINode):
        
        roi2RAS, roiOrigin_RAS = self.getBoxROIOriginCoordinates(markupBoxROINode)
        volumeNodeToTranslate.SetIJKToRASDirections(roi2RAS.GetElement(0,0), roi2RAS.GetElement(0,1), roi2RAS.GetElement(0,2), roi2RAS.GetElement(1,0), roi2RAS.GetElement(1,1), roi2RAS.GetElement(1,2), roi2RAS.GetElement(2,0), roi2RAS.GetElement(2,1), roi2RAS.GetElement(2,2))
        volumeNodeToTranslate.SetOrigin(roiOrigin_RAS[0:3])

        return volumeNodeToTranslate
    
    
    def fitBoxROImarkupToVolume(self, referenceVolumeNode, markupROINode):
        
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        cropVolumeParameters.SetInputVolumeNodeID(referenceVolumeNode.GetID())
        cropVolumeParameters.SetROINodeID(markupROINode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(cropVolumeParameters)  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)        
        
    # JU - End of user-defined section

    def process(self,
                inputVolumeSequenceNode: vtkMRMLSequenceNode, #vtkMRMLScalarVolumeNode,
                maskVolumeSegmentationNode: vtkMRMLSegmentationNode, #vtkMRMLScalarVolumeNode,
                outputMapsSequenceNode: vtkMRMLSequenceNode,
                outputLabelMapVolumeNode: vtkMRMLLabelMapVolumeNode,
                referenceBoxROINode: vtkMRMLMarkupsROINode, # Add roi Box as input argument
                serMapDictionary: dict,
                cadMapDictionary: dict,
                colourTableNode: vtkMRMLColorTableNode, # JU 08/07/2025 - Colour Table Node with the legend so can be updated with the results
                listOfNodesWithOmitRegions: list=[], # if empty, there is no omit regions to process
                tableNodeDict: dict={'TableName': [vtkMRMLTableNode, 'label_list']}, #vtkMRMLTableNode,
                preContrastIndex: int=0,
                earlyPostContrastIndex: int=1,
                latePostContrastIndex: int=-1,
                timings: dict={},
                PEthreshold: float=70.0,
                BKGRNDthreshold: float=60.0,
                segmentNodeID: str='',
                serUpperThreshold: float=3.0,
                useSERmap: bool=True
                ) -> None:
        """
        TODO: Update description of the input parameters
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolumeSequenceNode or not maskVolumeSegmentationNode:
            raise ValueError("Input or output volume is invalid")
        
        if preContrastIndex == earlyPostContrastIndex:
            raise ValueError("Pre Contrast Index Cannot be the same as the Early Post Contrast")
        
        if earlyPostContrastIndex == latePostContrastIndex:
            
            ok = slicer.util.confirmYesNoDisplay("Early and Late Post Contrast indices are the same. Shall we continue?", windowTitle="WARNING")
            if not ok:
                return

        # Get input volume dimensions
        inputVolume4Darray = self.getVolumeDataFromSequence(inputVolumeSequenceNode)
        [nt, nz, nx, ny] = inputVolume4Darray.shape

        # Allocate space in TICtable for the intensity values from the DCE array
        time_intensity_curve = np.full((nt, len(tableNodeDict['TICTable'][1])), np.nan)
        time_intensity_curve[:,0] = np.linspace(0, nt, nt, endpoint=False) 
        if timings is not None:
            time_intensity_curve[:, 0] = timings['timepoints'] / (1000 * 60) # From ms to min
        
        # JU - Create temporary volumes to work with inside this function:
        # Pre-populate it with the info from the first input volume in the input sequence, so we get the same image orientation,dimensions, etc.:
        tempReferenceVolumeNode = slicer.modules.volumes.logic().CloneVolume(inputVolumeSequenceNode.GetNthDataNode(0), "temporary")        
        tempSERVolumeNode = slicer.modules.volumes.logic().CloneVolume(inputVolumeSequenceNode.GetNthDataNode(0), "serMap")        
        tempPEVolumeNode  = slicer.modules.volumes.logic().CloneVolume(inputVolumeSequenceNode.GetNthDataNode(0), "peMap")
        # JU 01/05/2025 Allocate mem for the Washout Type map & add a label map that can later on be deleted if not used:
        tempWOUTVolumeNode  = slicer.modules.volumes.logic().CloneVolume(inputVolumeSequenceNode.GetNthDataNode(0), "woMap")
        # cadLabelMapVolumeNode  = slicer.modules.volumes.logic().CloneVolume(outputLabelMapVolumeNode, "ENH Label")        
        
        roiIJK = self.getBoxROIIJKCoordinates(referenceBoxROINode, tempReferenceVolumeNode)

        # MIP to be used as the backgdround image for the maps and set up a global threshold from the pre-contrast image
        mip_volume = np.max(inputVolume4Darray, axis=0)
        slicer.util.updateVolumeFromArray(tempReferenceVolumeNode, mip_volume)
        outputMapsSequenceNode.SetDataNodeAtValue(tempReferenceVolumeNode, "MIP")
        
        SERmapTemplate = np.zeros((nz,ny,nx))
        PEmapTemplate  = np.zeros((nz,ny,nx))
        # JU 01/05/2025: Add template for the Washout type, delta = (Sl-Se)/Se [%]
        DeltaMapTemplate = np.zeros((nz, ny, nx))

        # Get the segment selected by the list "Segment Label Mask":
        maskSegmentation = maskVolumeSegmentationNode.GetSegmentation()
         
        # Before doing anything, loop over the segmentation list and delete all labels created locally by this function in previous runs:
        if useSERmap:
            self.resetSegmentList(maskVolumeSegmentationNode, items_to_remove=serMapDictionary['legend'])
        else:
            # Do it again, but for the cadMapDictionary
            self.resetSegmentList(maskVolumeSegmentationNode, items_to_remove=cadMapDictionary['legend'])
        
        # Ensure visibility for the selected segment is ON (TRUE)
        selectedSegment = maskSegmentation.GetSegment(segmentNodeID)
        maskDisplayNode = maskVolumeSegmentationNode.GetDisplayNode()
        maskDisplayNode.SetSegmentVisibility(segmentNodeID, True)
        # Load the label map and check whether is empty or not. If empty, then use the ROI markup as the label map:
        label = slicer.util.arrayFromSegmentBinaryLabelmap(maskVolumeSegmentationNode, segmentNodeID)
        if not label.any():
            # the selected mask is empty --> use the ROI markup box only
            # Get ROI box as nd binary array:
            voi_mask = np.zeros((nz, ny, nx))
            voi_mask[roiIJK['IJKmin'][2]:roiIJK['IJKmax'][2], roiIJK['IJKmin'][1]:roiIJK['IJKmax'][1], roiIJK['IJKmin'][0]:roiIJK['IJKmax'][0]] = 1

            if listOfNodesWithOmitRegions:
                for omitRegionNode in listOfNodesWithOmitRegions:
                    omitRegIJK = self.getBoxROIIJKCoordinates(omitRegionNode, tempReferenceVolumeNode)
                    voi_mask[omitRegIJK['IJKmin'][2]:omitRegIJK['IJKmax'][2], omitRegIJK['IJKmin'][1]:omitRegIJK['IJKmax'][1], omitRegIJK['IJKmin'][0]:omitRegIJK['IJKmax'][0]] = 0
                    
            slicer.util.updateSegmentBinaryLabelmapFromArray(voi_mask, maskVolumeSegmentationNode, segmentNodeID)
            label = slicer.util.arrayFromSegmentBinaryLabelmap(maskVolumeSegmentationNode, segmentNodeID)
            selectedSegment.SetName('Segment from ROI')
                
        # Crop the volumes before doing any calculation 
        # TODO: For some unknown reason, even when cropping "manually" using the indices, the output maps get rotated around the ROI box corner (apparently)
        #       Fortunately, the time to process the whole volume is not that critical at this stage, so can live without cropping
        croppingVolumes = True
        if croppingVolumes:
            # Crop the volumes using the ROI box:
            # label = self.cropVolumeFromROI(label, referenceBoxROINode, tempReferenceVolumeNode)
            # inputVolume4Darray = self.cropSequenceVolumeFromROI(inputVolume4Darray, referenceBoxROINode, tempReferenceVolumeNode)
            inputVolume4Darray = inputVolume4Darray[:, roiIJK['IJKmin'][2]:roiIJK['IJKmax'][2], roiIJK['IJKmin'][1]:roiIJK['IJKmax'][1], roiIJK['IJKmin'][0]:roiIJK['IJKmax'][0]]
            label = label[roiIJK['IJKmin'][2]:roiIJK['IJKmax'][2], roiIJK['IJKmin'][1]:roiIJK['IJKmax'][1], roiIJK['IJKmin'][0]:roiIJK['IJKmax'][0]]
        
        # Represent the data in terms of SER (S(t)/S0(t)). identifying S0 as the pre-contrast index:
        St0 = inputVolume4Darray[preContrastIndex, :, :, :]
        # JU - add variables denoting the matrix size as they will be useful later in the code
        [nrows, ncols, ndepth] = St0.shape
         
        # The background threshold is defined from the masked section only:
        bckgrnd_thresh = (BKGRNDthreshold/100.0) * np.percentile(St0, 95)

        base_mask = (St0 >= bckgrnd_thresh) &  label

        St_minus_St0 = inputVolume4Darray - St0
        
        St1_minus_St0 = St_minus_St0[earlyPostContrastIndex, :, :, :]
        Stn_minus_St0 = St_minus_St0[latePostContrastIndex, :, :, :]
        
        PE = 100 * St1_minus_St0 / ( St0 + self.EPSILON )
        ETVthreshold = 100 * St1_minus_St0 / ( St0 + self.EPSILON )
        base_mask &= (PE >= PEthreshold)

        PE = np.where(base_mask, PE, 0)
        
        PEmapTemplate[roiIJK['IJKmin'][2]:roiIJK['IJKmax'][2], roiIJK['IJKmin'][1]:roiIJK['IJKmax'][1], roiIJK['IJKmin'][0]:roiIJK['IJKmax'][0]] = PE

        slicer.util.updateVolumeFromArray(tempPEVolumeNode, PEmapTemplate)
        outputMapsSequenceNode.SetDataNodeAtValue(tempPEVolumeNode, "PE")
        # Delete tempPEVolumeNode asap:
        slicer.mrmlScene.RemoveNode(tempPEVolumeNode)
        
        SER = ( St1_minus_St0 / ( Stn_minus_St0 + self.EPSILON ) ) 
        SER[SER < 0.0] = 0.0
        SER[SER > serUpperThreshold] = 0.0
        base_mask &= (SER >= 0.0)

        SER = np.where(base_mask, SER, 0)
                
        # JU - This convolution defines the maximum over a neighbourhood. But, it is not what is suppossed to do, according to the 
        #       reference literature.
        #       As defined in the main references (see e.g. Arasu et al. 2011, Partridge et al. 2010, and Xiao et al. 2021),
        #       peak PE and SER are defined as the highest mean over a 3x3x3 neighbourhood (or equivalently 8 contigous voxels).
        #       Moreover, this is just to summarise the results, so it shouldn't be used for display purposes, instead, it is used at the end,
        #       when reporting the results in the table. 
        # kernel = np.ones((3,3,3))
        # kernel[1,1,1] = 100
        # convbrmask = signal.convolve(base_mask, kernel, mode='same')
        # base_mask &= (convbrmask >= (100 + self.PIXEL_CONNECTIVITY))

        # Relevant for when adding a user-defined segmentation mask (e.g. Tumour_tissue)
        seg_points = np.where(base_mask)

        SERmap = np.zeros_like(SER)
        nLevels = len(serMapDictionary['levelThreshold']['UB'])
        for idx, (lb, ub) in enumerate(zip(serMapDictionary['levelThreshold']['LB'], serMapDictionary['levelThreshold']['UB'])):
            truth_table_indices = ( (SER > lb) & (SER <= ub) )
            SERmap[truth_table_indices] = idx + 1
            
        # Add the last element of the interval that makes MaxSER < SER:
        # JU 30/07/2024: How to deal with this if is NON-SER??
        idx = nLevels
        if idx == 0:
            SERmap[SER > 0.0] = 1

        SERmap *= base_mask
        
        SERmapTemplate[roiIJK['IJKmin'][2]:roiIJK['IJKmax'][2], roiIJK['IJKmin'][1]:roiIJK['IJKmax'][1], roiIJK['IJKmin'][0]:roiIJK['IJKmax'][0]] = SERmap
        
        slicer.util.updateVolumeFromArray(tempSERVolumeNode, SERmapTemplate)

        volumes_logic = slicer.modules.volumes.logic()
        if useSERmap:
            # Uncomment when enabling boolean flag for SER and CAD:
            volumes_logic.CreateLabelVolumeFromVolume(slicer.mrmlScene, outputLabelMapVolumeNode, tempSERVolumeNode)
            
            # Import all labels maps from outputLabelMapVolumeNode node to maskVolumeSegmentationNode, each label to a separate segment.
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputLabelMapVolumeNode, maskVolumeSegmentationNode)
            # End of branch 

        outputMapsSequenceNode.SetDataNodeAtValue(tempSERVolumeNode, "SER")
        
        # JU 01/05/2025: Adding Washout Type, WODELTA = 100 * (SLate - SEarly) / SEarly
        WODELTA =  100.0 * ( Stn_minus_St0 - St1_minus_St0 ) / (St1_minus_St0 + self.EPSILON)
        WASHOUTmap = np.zeros_like(WODELTA)

        for idx, label in enumerate(cadMapDictionary['legend']):
            if label == 'Plateau':
                truth_table_indices = np.abs(WODELTA) <= cadMapDictionary['levelThreshold']
            elif label == 'Persistent':
                truth_table_indices = WODELTA > cadMapDictionary['levelThreshold']
            elif label == 'Washout':
                truth_table_indices = WODELTA < ( -1.0 * cadMapDictionary['levelThreshold'] )
            else:
                # Everything else must be 0 already
                pass
            WASHOUTmap[truth_table_indices] = idx # The logic is different to SER maps, so in this case IDX is correct (instead of IDX+1)
        
        WASHOUTmap *= base_mask
        DeltaMapTemplate[roiIJK['IJKmin'][2]:roiIJK['IJKmax'][2], roiIJK['IJKmin'][1]:roiIJK['IJKmax'][1], roiIJK['IJKmin'][0]:roiIJK['IJKmax'][0]] = WASHOUTmap

        slicer.util.updateVolumeFromArray(tempWOUTVolumeNode, DeltaMapTemplate)
        if not useSERmap:
            # Uncomment when enabling boolean flag for SER and CAD:
            volumes_logic.CreateLabelVolumeFromVolume(slicer.mrmlScene, outputLabelMapVolumeNode, tempWOUTVolumeNode)
            
            # Import all labels maps from outputLabelMapVolumeNode node to maskVolumeSegmentationNode, each label to a separate segment.
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputLabelMapVolumeNode, maskVolumeSegmentationNode)
            # End of branch 

        # DEBUG
        AuxMaskSegmentation = maskVolumeSegmentationNode.GetSegmentation()
        for AuxSegment_iID in AuxMaskSegmentation.GetSegmentIDs():
            AuxSegmentName = AuxMaskSegmentation.GetSegment(AuxSegment_iID).GetName() #?
            print(f'Debuggin Line 1793 segment Name: {AuxSegmentName}')
        # END DEBUG block
        
        outputMapsSequenceNode.SetDataNodeAtValue(tempWOUTVolumeNode, "Washout")

        # Delete tempWOUTVolumeNode asap:
        slicer.mrmlScene.RemoveNode(tempWOUTVolumeNode)

        # JU 27/09/2024 - Here we calculated the peak PE and SER. (Partridge et al., 2010)
        # First, we find the mean over a 3x3x3 neighbourhood, 
        # and then get the max over them so we end up with a single value representing the peak PE and SER, 
        # respectively:
        mean_conv = np.ones((3,3,3))
        mean_conv /= mean_conv.sum()
        meanSERmap = signal.convolve(SER, mean_conv, mode='same')
        meanPEmap = signal.convolve(PE, mean_conv, mode='same')
        # Note that the convolution method to average a neighbourhood considers the values 0 when averaging (i.e. a=[1,0,1] ==> avg(a)=2/3)
        meanSERmap = meanSERmap[1:(nrows-np.mod(nrows,3)):3, 
                                1:(ncols-np.mod(ncols,3)):3, 
                                1:(ndepth-np.mod(ndepth,3)):3] # This retains only the average over the 3x3x3 sub-matrix (i.e. where the kernel fits complete in the volume)
        meanPEmap = meanPEmap[1:(nrows-np.mod(nrows,3)):3, 
                                1:(ncols-np.mod(ncols,3)):3, 
                                1:(ndepth-np.mod(ndepth,3)):3]
        peakSER = meanSERmap.max()
        peakPE = meanPEmap.max()
        
        if useSERmap:
            # FTV map label from SERmap:
            mapVolumes = {'FTV0': np.where(SER > 0.0, 1.0, 0.0),
                        'FTVthresh': np.where(SER > serMapDictionary['SERthreshold'], 1.0, 0.0),
                        # ETV is defined as the total number of pixels that enhance around 2 min post contrast injection 
                        # (Henderson et al., 2018), (Panthi et al., 2023)
                        'ETV':  np.where(ETVthreshold > 0.0, 1.0, 0.0) , # np.where(PE > 0, 1.0, 0.0),
                        }
        else:
            # JU (10/07/2025): FTV map label from CAD-like map
            #   Unlike SER, in this case, the FTV must be calculated differently
            #   It is only needed to sum over the actual ranges (not 0), so have to use WASHOUTmap
            mapVolumes = {'FTV0': np.where( WASHOUTmap > 0.0, 1.0, 0.0),
                        'FTVthresh': np.where( ( WASHOUTmap == cadMapDictionary['legend'].index('Plateau') ) | 
                                           ( WASHOUTmap == cadMapDictionary['legend'].index('Washout') ),
                                           1.0, 
                                           0.0),
                        #   ETV is defined as the total number of pixels that enhance around 2 min post contrast injection 
                        #   (Henderson et al., 2018), (Panthi et al., 2023)
                          'ETV': np.where(ETVthreshold > 0, 1.0, 0.0)
                          }
            
        
        mapStats = {}
        for mapNameID, mapVolume in mapVolumes.items():
            labelMapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "mapLabel")
            slicer.util.updateVolumeFromArray(tempSERVolumeNode, mapVolume)
            volumes_logic.CreateLabelVolumeFromVolume(slicer.mrmlScene, labelMapVolumeNode, tempSERVolumeNode)
            maskVolumeSegmentationNode.GetSegmentation().AddEmptySegment(mapNameID)
            mapSegmentID = vtk.vtkStringArray()
            mapSegmentID.InsertNextValue(mapNameID)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelMapVolumeNode, maskVolumeSegmentationNode, mapSegmentID)
            mapStats[mapNameID] = self.getStatsFromMask(maskVolumeSegmentationNode,
                                                        mapNameID)
            maskVolumeSegmentationNode.RemoveSegment(mapNameID)
            slicer.mrmlScene.RemoveNode(labelMapVolumeNode)

        slicer.mrmlScene.RemoveNode(tempSERVolumeNode)

        # JU - This operates over the Selected ROI (e.g. Tumour Tissue)
        uptake_ti = 100 * St_minus_St0 / (St0 + self.EPSILON)
        for time_index in range(nt):
            ser_roi  = uptake_ti[time_index,:,:,:]
            time_intensity_curve[time_index,1] =  ser_roi[seg_points].mean()

        max_ENH = np.max( uptake_ti, axis=0 )
        delta_ENH = ( uptake_ti[latePostContrastIndex,:,:,:] - uptake_ti[earlyPostContrastIndex,:,:,:] )[seg_points]
        first_pass_ENH = uptake_ti[earlyPostContrastIndex,:,:,:][seg_points]
        [m_slope, n_coeff], time_intensity_curve[1:,2] = self.simple_linear_fit(time_intensity_curve[1:,0], time_intensity_curve[1:,1])

        # Statistics for the user-defined Segmentation mask
        segmentStats = self.getStatsFromMask(maskVolumeSegmentationNode, segmentNodeID)
        # # Segment Oriented Bounding Box Diameter 
        maxROIDiameter = {'name': 'ROI longest axis',
                            'value': np.max(segmentStats['obb_diameter_mm']['value']),
                            'units': segmentStats['obb_diameter_mm']['units']}

        # ROI Volume:
        # The value returned by segmentStats is the enclosing box (Vol = prod(OBB_diameter)). 
        # To get the equivalent ellipsoidal volume, the formula is 4pi/3 * prod(OBB_radius)
        # Therefore, we need to multiply Vol by: (0.5)^3 * (4pi/3) (or simply pi/6.0)
        ellipsoidScale = (1.0/6.0) * np.pi
        roiVolume = {'name': 'ROI Volume',
                        'value': segmentStats['volume_cm3']['value'] * ellipsoidScale,
                        'units': segmentStats['volume_cm3']['units']}

        labelColumn = vtk.vtkStringArray()
        labelColumn.SetName(tableNodeDict['SummaryTable'][1][0])
        statsColumn = vtk.vtkDoubleArray()
        statsColumn.SetName(tableNodeDict['SummaryTable'][1][1])
        unitsColumn = vtk.vtkStringArray()
        unitsColumn.SetName(tableNodeDict['SummaryTable'][1][2])

        # Add stats to Summary Table:
        # JU 27/09/2024  - Add the peak PE and SER values at the begining of the table
        labelColumnContent = ['Peak SER',
                              'Peak PE',
                              'PE Threshold',
                              'SER Upper Threshold',
                              maxROIDiameter['name'], 
                              roiVolume['name'],
                              'Bolus injection time',
                              'Early Phase Time',
                              'Late Phase Time',
                              'Maximum Enhancement',
                              'Delta Enhancement',
                              'First Pass Enhancement',
                              'Enhancement Slope']

        statsColumnContent = [np.round(peakSER,3),
                              np.round(peakPE,3),
                              PEthreshold,
                              serMapDictionary['SERthreshold'],
                              maxROIDiameter['value'], 
                              roiVolume['value'],
                              timings['injectionTime']/(1000*60),
                              time_intensity_curve[earlyPostContrastIndex, 0],
                              time_intensity_curve[latePostContrastIndex, 0],
                              max_ENH[seg_points].mean(), 
                              delta_ENH.mean(), 
                              first_pass_ENH.mean(), 
                              m_slope]

        unitsColumnContent = ['[]',
                              '%',
                              '%',
                              '[]',
                              maxROIDiameter['units'], 
                              roiVolume['units'], 
                              'min',
                              'min',
                              'min',
                              '%', 
                              '%', 
                              '%', 
                              '[]']

        for rows in zip(labelColumnContent, statsColumnContent, unitsColumnContent):
            labelColumn.InsertNextValue(rows[0])
            statsColumn.InsertNextValue(rows[1])
            unitsColumn.InsertNextValue(rows[2])

        # Statistics for the SER Label Maps:
        nameColumn = vtk.vtkStringArray()
        nameColumn.SetName(tableNodeDict['SERSummaryTable'][1][0])
        volumeColumn = vtk.vtkDoubleArray()
        volumeColumn.SetName(tableNodeDict['SERSummaryTable'][1][1])
        distColumn = vtk.vtkDoubleArray()
        distColumn.SetName(tableNodeDict['SERSummaryTable'][1][2])

        # Iterate over the segmentation mask and get statistics for each SER label:
        maskSegmentations = maskVolumeSegmentationNode.GetSegmentation()
        
        FTVstats = [mapStats['FTV0']['volume_cm3']['value'], mapStats['FTV0']['voxel_count']['value']]
        FTVthresh_stats = [mapStats['FTVthresh']['volume_cm3']['value'], mapStats['FTVthresh']['voxel_count']['value']]
        ETVstats = [mapStats['ETV']['volume_cm3']['value'], mapStats['ETV']['voxel_count']['value']]

        if useSERmap: # SER ranges
            MapDictionary = serMapDictionary.copy()
            labelTitle = 'SER Ranges'
        else:
            MapDictionary = cadMapDictionary.copy()
            labelTitle = 'Labels'
            
        # Create summary table and legend for SER maps
        new_legend = list(MapDictionary['colourMap'].keys())
        new_legend_cmap = list(MapDictionary['colourMap'].values())
        
        for segment_iID in maskSegmentations.GetSegmentIDs():
            segmentName = maskSegmentation.GetSegment(segment_iID).GetName() #?
            if segmentName in MapDictionary['legend']: #SERauxList:
                segmentPos = MapDictionary['legend'].index(segmentName)# SERauxList.index(segmentName)
                segmentStats = self.getStatsFromMask(maskVolumeSegmentationNode, segment_iID)
                vol_cm3 = np.round(segmentStats['volume_cm3']['value'],3)
                vol_rel = np.round(100 * segmentStats['voxel_count']['value'] / FTVstats[1], 2)
                nameColumn.InsertNextValue(segmentName)
                volumeColumn.InsertNextValue(vol_cm3)
                distColumn.InsertNextValue(vol_rel)
                new_legend[segmentPos] = f'{segmentName}\n{vol_rel:.2f}%'
                # to ensure new_legend and colour_map position matches:
                new_legend_cmap[segmentPos] = MapDictionary['colourMap'][segmentName]
        
        updated_legend = {
            'ColourNode': colourTableNode,
            'dict_legend': dict(zip(new_legend, new_legend_cmap)),
            'Title': labelTitle
            }
            
        
        # Append the FTV and ETV stats at the end of list
        nameColumn.InsertNextValue('FTVth (SER>SER_th)')
        volumeColumn.InsertNextValue(np.round(FTVthresh_stats[0],3))
        distColumn.InsertNextValue(np.round(100 * FTVthresh_stats[1]/FTVstats[1], 2))

        nameColumn.InsertNextValue('FTV (SER>0)')
        volumeColumn.InsertNextValue(np.round(FTVstats[0],3))
        distColumn.InsertNextValue(np.round(100 * FTVstats[1]/FTVstats[1], 2))

        nameColumn.InsertNextValue('ETV (Enhanced Tumour Volume)')
        volumeColumn.InsertNextValue(np.round(ETVstats[0],3))
        distColumn.InsertNextValue(np.round(100 * ETVstats[1]/ETVstats[1], 2))
         
        # JU - Update table and plot - TODO: I think this should be moved to a different function
        slicer.util.updateTableFromArray(tableNodeDict['TICTable'][0], time_intensity_curve, tableNodeDict['TICTable'][1])

        tableNodeDict['SummaryTable'][0].AddColumn(labelColumn)
        tableNodeDict['SummaryTable'][0].AddColumn(statsColumn)
        tableNodeDict['SummaryTable'][0].AddColumn(unitsColumn)
        tableNodeDict['SERSummaryTable'][0].AddColumn(nameColumn)
        tableNodeDict['SERSummaryTable'][0].AddColumn(volumeColumn)
        tableNodeDict['SERSummaryTable'][0].AddColumn(distColumn)

        # Update viewer with results:
        updatedSequenceBrowserNode = self.findBrowserForSequence(outputMapsSequenceNode)
        slicer.modules.sequences.toolBar().setActiveBrowserNode(updatedSequenceBrowserNode)
        # Set background image to be the MIP:
        updatedSequenceBrowserNode.SetSelectedItemNumber(0) # MIP is the first node we added
        self.updateViewer(backgroundVolume=updatedSequenceBrowserNode.GetProxyNode(outputMapsSequenceNode),
                        labelVolume=outputLabelMapVolumeNode, legend_update=updated_legend)

        self.showVolumeRenderingMIP(updatedSequenceBrowserNode.GetProxyNode(outputMapsSequenceNode))
        # # Import label map into a segmentation:
        # slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputLabelMapVolumeNode, maskVolumeSegmentationNode)
        # Add the SER maps to the 3D rendering:
        maskVolumeSegmentationNode.CreateClosedSurfaceRepresentation()
        # slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(outputLabelMapVolumeNode)
        # Finally, remove the temporary nodes (it should be wrapped into a try/except/finally statement to ensure it always get deleted)
        slicer.mrmlScene.RemoveNode(tempReferenceVolumeNode)
        
        

# quantificationTest
#


# class quantificationTest(ScriptedLoadableModuleTest):
#     """
#     This is the test case for your scripted module.
#     Uses ScriptedLoadableModuleTest base class, available at:
#     https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
#     """

#     def setUp(self):
#         """Do whatever is needed to reset the state - typically a scene clear will be enough."""
#         slicer.mrmlScene.Clear()

#     def runTest(self):
#         """Run as few or as many tests as needed here."""
#         self.setUp()
#         self.test_quantification1()

#     def test_quantification1(self):
#         """Ideally you should have several levels of tests.  At the lowest level
#         tests should exercise the functionality of the logic with different inputs
#         (both valid and invalid).  At higher levels your tests should emulate the
#         way the user would interact with your code and confirm that it still works
#         the way you intended.
#         One of the most important features of the tests is that it should alert other
#         developers when their changes will have an impact on the behavior of your
#         module.  For example, if a developer removes a feature that you depend on,
#         your test should break so they know that the feature is needed.
#         """

#         self.delayDisplay("Starting the test")

#         # Get/create input data

#         import SampleData

#         registerSampleData()
#         inputVolume = SampleData.downloadSample("quantification1")
#         self.delayDisplay("Loaded test data set")

#         inputScalarRange = inputVolume.GetImageData().GetScalarRange()
#         self.assertEqual(inputScalarRange[0], 0)
#         self.assertEqual(inputScalarRange[1], 695)

#         outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
#         threshold = 100

#         # Test the module logic

#         logic = quantificationLogic()

#         # Test algorithm with non-inverted threshold
#         logic.process(inputVolume, outputVolume, threshold, True)
#         outputScalarRange = outputVolume.GetImageData().GetScalarRange()
#         self.assertEqual(outputScalarRange[0], inputScalarRange[0])
#         self.assertEqual(outputScalarRange[1], threshold)

#         # Test algorithm with inverted threshold
#         logic.process(inputVolume, outputVolume, threshold, False)
#         outputScalarRange = outputVolume.GetImageData().GetScalarRange()
#         self.assertEqual(outputScalarRange[0], inputScalarRange[0])
#         self.assertEqual(outputScalarRange[1], inputScalarRange[1])

#         self.delayDisplay("Test passed")
