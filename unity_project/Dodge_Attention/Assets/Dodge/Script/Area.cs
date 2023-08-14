using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class Area : MonoBehaviour
{
    public GameObject Agent;
    public GameObject Ball;
    public GameObject Env;
    public GameObject Wall;

    List<GameObject> balls = new List<GameObject>();
    List<BallScript> ballScripts = new List<BallScript>();

    public float boardRadius = 6.0f;
    public float ballSpeed = 3.0f;
    public int ballNums = 15;
    public float ballRandom = 0.2f;
    public float agentSpeed = 3.0f;

    DodgeAgent agentScript = null;
    EnvironmentParameters m_ResetParams = null;

    void Start()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        agentScript = Agent.GetComponent<DodgeAgent>();
        InitBall(); 
    }

    public void InitBall()
    {
        ballScripts.Clear();

        for (int i = 0; i < balls.Count; i++)
            Destroy(balls[i]);

        balls.Clear();

        for (int i = 0; i < ballNums; i++)
        {
            GameObject b = Instantiate(Ball, Env.transform);
            BallScript script = b.GetComponent<BallScript>();
            script.SetBall(Agent, ballSpeed, ballRandom, boardRadius);
            balls.Add(b);
            ballScripts.Add(script);
        }
    }

    public void ResetEnv()
    {
        if (null == m_ResetParams)
            m_ResetParams = Academy.Instance.EnvironmentParameters;

        boardRadius = m_ResetParams.GetWithDefault("boardRadius", boardRadius);
        ballSpeed = m_ResetParams.GetWithDefault("ballSpeed", ballSpeed);
        ballNums = (int)m_ResetParams.GetWithDefault("ballNums", ballNums);
        ballRandom = m_ResetParams.GetWithDefault("ballRandom", ballRandom);
        agentSpeed = m_ResetParams.GetWithDefault("agentSpeed", agentSpeed);

        if (null == agentScript)
            agentScript = Agent.GetComponent<DodgeAgent>();

        agentScript.SetAgentSpeed(agentSpeed);
        Wall.transform.localScale = new Vector3(boardRadius, 10f, boardRadius);

        InitBall(); 
    }
}
