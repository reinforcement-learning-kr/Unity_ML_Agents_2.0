using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class pongArea : MonoBehaviour
{
    [SerializeField]
    private PongAgent agentA = null;
    [SerializeField]
    private PongGoalDetection goalDetection = null;

    public void Start()
    {
        if (null != agentA)
            agentA.SetDelegate(OnEpisodeBegin);
    }

    public void OnEpisodeBegin()
    {
        if (null != goalDetection)
            goalDetection.ResetPostion();
    }
}
