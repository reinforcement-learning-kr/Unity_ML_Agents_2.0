using UnityEngine;
using Unity.InteractiveTutorials;
using UnityEditor;

namespace Unity.Tutorials
{
    /// <summary>
    /// Implement your Tutorial callbacks here.
    /// </summary>
    public class TutorialCallbacks : ScriptableObject
    {
        public FutureObjectReference futureJumpInstance = default;
        public FutureObjectReference futureBotInstance = default;

        /// <summary>
        /// Keeps the Jump selected during a tutorial. 
        /// </summary>
        public void KeepJumpSelected()
        {
            SelectSpawnedGameObject(futureJumpInstance);
        }

        /// <summary>
        /// Keeps the Bot selected during a tutorial. 
        /// </summary>
        public void KeepBotSelected()
        {
            SelectSpawnedGameObject(futureBotInstance);
        }


        /// <summary>
        /// Selects a GameObject in the scene, marking it as the active object for selection
        /// </summary>
        /// <param name="futureObjectReference"></param>
        public void SelectSpawnedGameObject(FutureObjectReference futureObjectReference)
        {
            if (futureObjectReference.sceneObjectReference == null) { return; }
            Selection.activeObject = futureObjectReference.sceneObjectReference.ReferencedObjectAsGameObject;
        }

        public void SelectMoveTool()
        {
            Tools.current = Tool.Move;
        }

        public void SelectRotateTool()
        {
            Tools.current = Tool.Rotate;
        }
    }
}