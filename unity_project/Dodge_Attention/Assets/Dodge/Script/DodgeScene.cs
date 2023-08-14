using UnityEngine;
using Unity.MLAgents;

public class DodgeScene : MonoBehaviour
{
    public GameObject areaObj = null;

    private int AreaNums = 9;
    private int rows = 3;
    private int interval = 16;

    EnvironmentParameters m_ResetParams = null; 

    void Start()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        AreaNums = (int)m_ResetParams.GetWithDefault("AreaNums", AreaNums);

        float x = 0;
        float z = 0; 

        for (int i = 0; i < AreaNums; i++)
        {
            GameObject areaPrefab = Instantiate(areaObj, transform); 

            if (i % rows == 0 && i != 0)
            {
                x = 0;
                z += interval;
            }

            areaPrefab.transform.position = new Vector3(x, 0, z);
            x += interval; 
        }
    }
}
