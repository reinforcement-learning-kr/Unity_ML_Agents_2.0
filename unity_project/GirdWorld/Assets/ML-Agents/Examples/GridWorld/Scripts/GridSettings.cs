using UnityEngine;
using Unity.MLAgents;

public class GridSettings : MonoBehaviour
{
    public Camera MainCamera;


    // Unity 내장함수로써 프로젝트가 시작될 때 한번만 호출되며 Main Camera의 위치, 각도를 알맞게 조절한다.
    public void Awake()
    {
        Academy.Instance.EnvironmentParameters.RegisterCallback("gridSize", f =>
        {
            MainCamera.transform.position = new Vector3(-(f - 1) / 2f, f * 1.25f, -(f - 1) / 2f);
            MainCamera.orthographicSize = (f + 5f) / 2f;
        });

    }
}
