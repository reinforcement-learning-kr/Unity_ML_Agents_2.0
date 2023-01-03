using System;
using UnityEngine;


public enum GoalType
{
    WallPass = 0,
    PushBall = 1
}

public class TwoMissionsEnvController : MonoBehaviour
{
    public TwoMissionsAgent agent;
    public GameObject ball;
    public GameObject wallPassGoal;
    public GameObject pushBallGoal;
    public GameObject agentSpawnArea;
    public GameObject ballSpawnArea;


    public Material goalMaterial;
    public Material nonGoalMaterial;
    public Transform leftWallTr;
    public Transform rightWallTr;

    private Renderer pushBallRenderer;
    private Renderer wallPassRenderer;
    private Rigidbody agentRb;
    private Rigidbody ballRb;
    private Bounds agentSpawnAreaBounds;
    private Bounds ballSpawnAreaBounds;
    private GoalType curGoal;
    private TwoMissionsGoalDetect ballGoalDetector;
    private float[] goalsOneHotArr;
    private const float WALL_ENTRANCE_LENGTH = 5.0f;
    private const float WALL_HEIGHT = 2.0f;
    private const float WALL_THICKNESS = 1.0f;
    private const float FLOOR_WIDTH = 20f;


    public float[] GoalsOneHotArr
    {
        get => goalsOneHotArr;
    }

    private const int GOAL_TOTAL_COUNT = 2;

    public GoalType CurrentGoal
    {
        get => curGoal;
    }

    public void InitializeEnv()
    {
        wallPassRenderer = wallPassGoal.GetComponent<Renderer>();
        pushBallRenderer = pushBallGoal.GetComponent<Renderer>();
        agentRb = agent.GetComponent<Rigidbody>();
        ballRb = ball.GetComponent<Rigidbody>();
        ballGoalDetector = ball.GetComponent<TwoMissionsGoalDetect>();
        ballGoalDetector.ReachedGoal.AddListener(OnBallGoalReached);

        agentSpawnAreaBounds = agentSpawnArea.GetComponent<Collider>().bounds;
        ballSpawnAreaBounds = ballSpawnArea.GetComponent<Collider>().bounds;
        agentSpawnArea.SetActive(false);
        ballSpawnArea.SetActive(false);

        ResetScene();
    }


    public void ResetScene()
    {
        curGoal = (GoalType)UnityEngine.Random.Range(0, GOAL_TOTAL_COUNT);

        // one hot encoding
        goalsOneHotArr = new float[GOAL_TOTAL_COUNT];
        Array.Clear(goalsOneHotArr, 0, goalsOneHotArr.Length);
        goalsOneHotArr[(int)curGoal] = 1;

        // set color to show the current goal
        switch (curGoal)
        {
            case GoalType.PushBall:
                wallPassRenderer.material = nonGoalMaterial;
                pushBallRenderer.material = goalMaterial;
                break;
            case GoalType.WallPass:
                wallPassRenderer.material = goalMaterial;
                pushBallRenderer.material = nonGoalMaterial;
                break;
            default:
                break;
        }

        // reset ball
        ballRb.velocity = Vector3.zero;
        ballRb.angularVelocity = Vector3.zero;
        ballRb.transform.position = GetRandomSpawnPos(ballSpawnAreaBounds, ballRb.transform.position.y);
        ballRb.drag = 0.5f;

        // reset agent
        agentRb.velocity = Vector3.zero;
        agentRb.angularVelocity = Vector3.zero;
        agent.transform.position = GetRandomSpawnPos(agentSpawnAreaBounds, agent.transform.position.y);

        // reset wall
        float holePos = UnityEngine.Random.Range(
            WALL_ENTRANCE_LENGTH * 0.5f,
            FLOOR_WIDTH - WALL_ENTRANCE_LENGTH * 0.5f);
        leftWallTr.localScale = new Vector3(
            holePos - WALL_ENTRANCE_LENGTH * 0.5f,
            WALL_HEIGHT,
            WALL_THICKNESS);
        rightWallTr.localScale = new Vector3(
            FLOOR_WIDTH - holePos - WALL_ENTRANCE_LENGTH * 0.5f,
            WALL_HEIGHT,
            WALL_THICKNESS);


    }

    public Vector3 GetRandomSpawnPos(Bounds bounds, float yPos)
    {
        var randomPosX = UnityEngine.Random.Range
        (-bounds.extents.x,
            bounds.extents.x);
        var randomPosZ = UnityEngine.Random.Range(
            -bounds.extents.z,
            bounds.extents.z);

        var randomSpawnPos = bounds.center +
            new Vector3(randomPosX, yPos - bounds.center.y, randomPosZ);
        return randomSpawnPos;
    }

    private void OnBallGoalReached()
    {
        if (CurrentGoal == GoalType.PushBall)
        {
            Debug.Log("Ball goal reached!");
            agent.ReachedGoal();
        }
    }

    public bool IsAgentGround()
    {
        return Physics.Raycast(agent.transform.position, Vector3.down, 20);
    }

    public bool IsBallIsOnGround()
    {
        return Physics.Raycast(ballRb.position, Vector3.down, 20);
    }
}
