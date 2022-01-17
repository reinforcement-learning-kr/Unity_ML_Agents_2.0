
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class PongAgent : Agent
{
    public GameObject Ball = null;
    public GameObject Opponent = null;
    private Rigidbody RbBall = null;
    private Rigidbody RbAgent = null;
    private Rigidbody RbOpponent = null;

    private Transform agentTrans = null;
    private Transform OpponentTrans = null;
    private Transform ballTrans = null;

    public float timeBetweenDecisionsAtInference;
    float m_TimeSinceDecision;

    public delegate void OnEpisodeBeginDel();
    public OnEpisodeBeginDel onEpisodeBeginDel;

    public enum ActionType: int
    {
        NONE = -1,
        STAY,
        RIGHT,
        LEFT
    }

    private Vector3 resetPos = Vector3.zero;
    private Vector3 ballResetPos = Vector3.zero;

    public override void Initialize()
    {
        base.Initialize();

        RbAgent = gameObject.GetComponent<Rigidbody>();
        RbBall = Ball.GetComponent<Rigidbody>();
        RbOpponent = Opponent.GetComponent<Rigidbody>();

        agentTrans = transform;
        OpponentTrans = RbOpponent.transform;
        ballTrans = Ball.transform;

        resetPos = agentTrans.position;
        ballResetPos = ballTrans.position;

        Academy.Instance.AgentPreStep += WaitTimeInference;
    }

    public void AgentReset()
    {
        agentTrans.position = resetPos;
        RbAgent.velocity = Vector3.zero;
        RbAgent.angularVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // º»ÀÎ ÁÂÇ¥
        sensor.AddObservation(agentTrans.position.x);

        // »ó´ë¹æ ÁÂÇ¥
        sensor.AddObservation(OpponentTrans.position.x);

        // BallÀÇ ÁÂÇ¥
        sensor.AddObservation(ballTrans.position.x);
        sensor.AddObservation(ballTrans.position.z);

        // BallÀÇ ¼Ó·Â
        sensor.AddObservation(RbBall.velocity.x);
        sensor.AddObservation(RbBall.velocity.z);

    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        ActionType actions = (ActionType)actionBuffers.DiscreteActions[0];
        Vector3 targetPos = agentTrans.position;

        switch (actions)
        {
            case ActionType.STAY:
                targetPos = agentTrans.position;
                break;
            case ActionType.RIGHT:
                targetPos = agentTrans.position + Vector3.right;
                break;
            case ActionType.LEFT:
                targetPos = agentTrans.position + Vector3.left;
                break;
        }

        var hit = Physics.OverlapBox(targetPos, new Vector3(0.5f, 0.5f, 0.5f));

        if (hit.Where(col => col.gameObject.CompareTag("Wall")).ToArray().Length == 0)
            agentTrans.position = targetPos;
    }

    public void OpponentScored()
    {
        AddReward(-1f);
        EndEpisode();
    }

    public void ScoredGoal()
    {
        AddReward(1f);
        EndEpisode();
    }

    private void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.CompareTag("Wall"))
        {
            RbAgent.velocity = Vector3.zero;
            RbAgent.angularVelocity = Vector3.zero;
        }

        if(collision.gameObject.CompareTag("Ball"))
        {
            AddReward(0.5f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
    }

    void WaitTimeInference(int action)
    {
        if (Academy.Instance.IsCommunicatorOn)
        {
            RequestDecision();
        }
        else
        {
            if (m_TimeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                m_TimeSinceDecision = 0f;
                RequestDecision();
            }
            else
            {
                m_TimeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }

    public void SetDelegate(OnEpisodeBeginDel onEpisodeBeginDel_)
    {
        onEpisodeBeginDel = onEpisodeBeginDel_;
    }

    public override void OnEpisodeBegin()
    {
        if (null != onEpisodeBeginDel)
            onEpisodeBeginDel();
        
    }
}
