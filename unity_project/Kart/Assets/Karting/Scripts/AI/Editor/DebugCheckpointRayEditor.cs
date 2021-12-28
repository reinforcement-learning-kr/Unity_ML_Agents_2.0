using UnityEditor;
using UnityEngine;

namespace KartGame.AI.EditorTools
{
    /// <summary>
    /// Draws a custom inspector and scene view to assist when editing colliders to train your agents.
    /// </summary>
    [CustomEditor(typeof(DebugCheckpointRay))]
    public class DebugCheckpointRayEditor : Editor 
    {
        DebugCheckpointRay m_DebugCheckpointRay;

        void OnEnable()
        {
            m_DebugCheckpointRay = target as DebugCheckpointRay;
        }

        void OnSceneGUI()
        {
            Collider[] colliders = m_DebugCheckpointRay.Colliders;

            GUIStyle textStyle = new GUIStyle();
            textStyle.normal.textColor = Color.white;

            // Render labels offseted by the collider's x scale which shows the order of the collider.
            for (int i = 0; i < colliders.Length; i++)
            {
                Transform current = colliders[i].transform;
                Vector3 position = current.position + current.right * current.localScale.x / 2f;
                Handles.Label(position, $"Collider #{i + 1}", textStyle);
            }
        }

        void DrawRenamingButton()
        {
            if (GUILayout.Button($"Rename Agent Checkpoints"))
            {
                Collider[] colliders = m_DebugCheckpointRay.Colliders;
                for (int i = 0; i < colliders.Length; i++)
                {
                    colliders[i].name = $"{m_DebugCheckpointRay.ColliderNameTemplate} {i + 1}";
                }
            }
        }

        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();
            DrawRenamingButton();
        }
    }
}

