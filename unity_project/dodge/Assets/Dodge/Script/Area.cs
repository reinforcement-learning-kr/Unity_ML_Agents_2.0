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
    List<GameObject> balls = new List<GameObject>();
    List<BallScript> ballScripts = new List<BallScript>();

    [Header("--ResetParams--")]
    public float boardRadius = 6.0f;
    public float ballSpeed = 3.0f;
    public int ballNums = 15;
    public float ballRandom = 0.2f;
    public int randomSeed = 77;
    public float agentSpeed = 30f;

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
            GameObject b = Instantiate(Ball, Env.transform);
            BallScript script = b.GetComponent<BallScript>();
            script.SetBall(Agent, ballSpeed, ballRandom, boardRadius);
            balls.Add(b);
            ballScripts.Add(script);
            ballScripts[i].gameObject.SetActive(false);
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
        randomSeed = (int)m_ResetParams.GetWithDefault("randomSeed", randomSeed);
        agentSpeed = m_ResetParams.GetWithDefault("agentSpeed", agentSpeed);


        if (ballNums != ballScripts.Count)
        {
            ballScripts.Clear();

            for (int i = 0; i < balls.Count; i++)
                Destroy(balls[i]);

            balls.Clear();
            InitBall();
        }
        else
        {
            for (int i = 0; i < ballNums; i++)
            {
                if (null != ballScripts[i])
                    ballScripts[i].gameObject.SetActive(false);
            }
        }

        if (null == agentScript)
            agentScript = Agent.GetComponent<DodgeAgent>();

        agentScript.SetAgentSpeed(agentSpeed);
        Wall.transform.localScale = new Vector3(boardRadius, 10f, boardRadius);

        for (int i = 0; i < ballNums; i++)
        {
            if (null != ballScripts[i])
                ballScripts[i].SetBall(Agent, ballSpeed, ballRandom, boardRadius);

            if (null != ballScripts[i])
                ballScripts[i].gameObject.SetActive(true);
        }
    }
}
