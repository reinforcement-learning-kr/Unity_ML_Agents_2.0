using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class pongArea : MonoBehaviour
{
    public PongAgent agentA = null;
    public PongGoalDetection goalDetection = null;

    void Start()
    {
        agentA.SetDelegate(OnEpisodeBegin);
    }

    public void OnEpisodeBegin()
    {
        goalDetection.ResetPosition();
    }
}
