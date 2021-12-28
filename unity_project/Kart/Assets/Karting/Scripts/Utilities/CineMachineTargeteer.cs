using Cinemachine;
using UnityEngine;

namespace KartGame.Utilities
{
    /// <summary>
    /// A preview utility to help cycle the Cinemachine Camera to look at and follow various targets.
    /// </summary>
    [RequireComponent(typeof(CinemachineVirtualCamera))]
    public class CineMachineTargeteer : MonoBehaviour
    {
        [Tooltip("What gameObjects do we want to look at?")]
        public Transform[] Targets;
        [Tooltip("Which key allows us to switch targets? By default, we use the Spacebar.")]
        public KeyCode CyclingKey = KeyCode.Space;

        CinemachineVirtualCamera m_VirtualCam;
        int m_Index;

        void Start()
        {
            m_VirtualCam = GetComponent<CinemachineVirtualCamera>();
            m_Index = 0;
        }

        void Update()
        {
            if (Input.GetKeyDown(CyclingKey))
            {
                m_Index             = (m_Index + 1) % Targets.Length;
                Transform target    = Targets[m_Index];
                m_VirtualCam.Follow = m_VirtualCam.LookAt = target;

#if UNITY_EDITOR
                // By pinging the gameObject we can easily find the gameObject in the Hierarchy view.
                UnityEditor.EditorGUIUtility.PingObject(m_VirtualCam.Follow);
#endif
            }
        }
    }
}
