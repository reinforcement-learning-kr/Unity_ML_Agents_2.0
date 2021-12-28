using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

//Automatically lock the UI layer on load, for convenience when editing the level.
[InitializeOnLoad]
public class AutoLockUILayer
{

    //This runs when the Editor loads, and when scripts get compiled.
    static AutoLockUILayer(){
        //Listen for when the user's selection changes.
        UnityEditor.Selection.selectionChanged += OnSelectionChanged;
        //Lock UI layer to begin with.
        SetUILayerIsLocked(true);
    }

    //Every time the inspector selection changes, rescan and see if we should lock or unlock the UI layer.
    static void OnSelectionChanged(){
        var activeGameObject = UnityEditor.Selection.activeGameObject;
        if (activeGameObject != null && activeGameObject.layer == 5){
            //if the object we have selected is in the UI layer, unlock the UI layer.
            SetUILayerIsLocked(false);
        }
        else{
            //otherwise, lock the UI layer.
            SetUILayerIsLocked(true);
        }
    }

    static void SetUILayerIsLocked(bool active){
        //Create a layer mask for the UI layer
        var uiLayer = LayerMask.GetMask("UI");
        if (active){
            UnityEditor.Tools.lockedLayers |= uiLayer;
        }
        else{
            //Mask inversion
            uiLayer = ~uiLayer;
            UnityEditor.Tools.lockedLayers &= uiLayer;
        }
    }

}
