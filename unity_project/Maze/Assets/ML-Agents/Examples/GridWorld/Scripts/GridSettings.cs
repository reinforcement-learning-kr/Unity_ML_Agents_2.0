using UnityEngine;
using Unity.MLAgents;

public class GridSettings : MonoBehaviour
{
    public Camera MainCamera;

    public void Awake()
    {
        Academy.Instance.EnvironmentParameters.RegisterCallback("gridSize", f =>
        {
            MainCamera.transform.position = new Vector3(-(f - 1) / 2f, f * 1.25f, -(f - 1) / 2f);
            MainCamera.orthographicSize = (f + 5f) / 2f;
        });

    }
}
