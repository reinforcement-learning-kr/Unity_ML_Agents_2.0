using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class Area : MonoBehaviour
{
    public GameObject Agent;
    public GameObject Ball;
    public GameObject Env;
    public GameObject Wall;
    List<BallScript> ballScripts = new List<BallScript>();

    [Header("--ResetParams--")]
    private float boardRadius = 7.5f;
    private float ballSpeed = 3.0f;
    private int ballNums = 15;
    private float ballRandom = 0.2f;
    private int randomSeed = 77;
    private float agentSpeed = 30f;

    DodgeAgent agentScript = null;

    EnvironmentParameters m_ResetParams = null;

    private void Start()
    {
        m_ResetParams = Academy.Instance.EnvironmentParameters;
        agentScript = Agent.GetComponent<DodgeAgent>();
        InitBall();
    }

    public void InitBall()
    {
        for (int i = 0; i < ballNums; i++)
        {
            GameObject ball = Instantiate(Ball, Env.transform);
            BallScript script = ball.GetComponent<BallScript>();
            script.SetBall(Agent, ballSpeed, ballRandom, boardRadius);
            ballScripts.Add(script);
            script.gameObject.SetActive(false);
        }
    }

    public void ResetEnv()
    {
        if (ballScripts.Count == 0)
            InitBall();

        for (int i = 0; i < ballScripts.Count; i++)
            if (null != ballScripts[i])
                ballScripts[i].gameObject.SetActive(false);

        if (null == m_ResetParams)
            m_ResetParams = Academy.Instance.EnvironmentParameters;

        boardRadius = m_ResetParams.GetWithDefault("boardRadius", boardRadius);
        ballSpeed = m_ResetParams.GetWithDefault("ballSpeed", ballSpeed);
        ballNums = (int)m_ResetParams.GetWithDefault("ballNums", ballNums);
        ballRandom = m_ResetParams.GetWithDefault("ballRandom", ballRandom);
        randomSeed = (int)m_ResetParams.GetWithDefault("randomSeed", randomSeed);
        agentSpeed = m_ResetParams.GetWithDefault("agentSpeed", agentSpeed);

        if (null == agentScript)
            agentScript = Agent.GetComponent<DodgeAgent>();

        agentScript.SetAgentSpeed(agentSpeed);
        Wall.transform.localScale = new Vector3(boardRadius, 10f, boardRadius);

        for (int i = 0; i < ballScripts.Count; i++)
        {
            if(null != ballScripts[i])
            {
                ballScripts[i].SetBall(Agent, ballSpeed, ballRandom, boardRadius);
                ballScripts[i].gameObject.SetActive(true);
            }
        }
        
    }
}
