using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

public class HardExplorationAgent : Agent
{
    Rigidbody mAgentRb;
    public int keyState;

    public GameObject key1;
    public GameObject key2;
    public GameObject key3;

    public GameObject gate1;
    public GameObject gate2;

    public float speed = 9f;
    public float rotationSpeed = 450f;
    public float jumpForce = 5.2f;

    public GameObject walls;

    public bool canJump = false;

    public override void Initialize()
    {
        base.Initialize();
        mAgentRb = this.gameObject.GetComponent<Rigidbody>();
        Academy.Instance.AgentPreStep += WaitTimeInference;
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(20, 0.5f, -7.5f);
        transform.localRotation = Quaternion.Euler(0, 0, 0);
        mAgentRb.velocity = Vector3.zero;
        mAgentRb.angularVelocity = Vector3.zero;

        keyState = 0;

        key1.SetActive(true);
        key2.SetActive(true);
        key3.SetActive(true);

        gate1.SetActive(true);
        gate2.SetActive(true);
    }

    public void MoveAgent(ActionBuffers actionBuffers)
    {
        var discreteActions = actionBuffers.DiscreteActions;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        dirToGo = transform.forward * discreteActions[1];
        switch (discreteActions[2])
        {
            case 0:
                break;
            case 1:
                rotateDir = -transform.up;
                break;
            case 2:
                rotateDir = transform.up;
                break;
        }

        if (discreteActions[0] == 1 && canJump)
        {
            canJump = false;
            mAgentRb.AddForce(transform.up * jumpForce, ForceMode.VelocityChange);
        }

        mAgentRb.transform.Translate(dirToGo * speed * Time.fixedDeltaTime, Space.World);
        transform.Rotate(rotateDir, Time.fixedDeltaTime * rotationSpeed);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        MoveAgent(actions);

        switch (keyState)
        {
            case 0:
                {
                    if (Vector3.Distance(key1.transform.position, transform.position) <= 1.5f)
                    {
                        key1.SetActive(false);
                        gate1.SetActive(false);
                        keyState = 1;
                    }
                }
                break;
            case 1:
                {
                    if (Vector3.Distance(key2.transform.position, transform.position) <= 1.5f)
                    {
                        key2.SetActive(false);
                        gate2.SetActive(false);
                        keyState = 2;
                    }
                }
                break;
            case 2:
                {
                    if (Vector3.Distance(key3.transform.position, transform.position) <= 1.5f)
                    {
                        SetReward(1f);
                        EndEpisode();
                    }
                }
                break;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("ground"))
        {
            canJump = true;
        }

        if (collision.gameObject.CompareTag("stairs"))
        {
            canJump = true;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        discreteActionsOut[0] = Input.GetKey(KeyCode.Space) ? 1 : 0;
        discreteActionsOut[1] = Input.GetKey(KeyCode.W) ? 1 : 0;

        if (Input.GetKey(KeyCode.Q))
        {
            discreteActionsOut[2] = 1;
        }
        if (Input.GetKey(KeyCode.E))
        {
            discreteActionsOut[2] = 2;
        }
    }

    float DecisionWaitingTime = 0.02f;
    float m_currentTime = 0f;

    public void WaitTimeInference(int action)
    {
        if (Academy.Instance.IsCommunicatorOn)
        {
            RequestDecision();
        }
        else
        {
            if (m_currentTime >= DecisionWaitingTime)
            {
                m_currentTime = 0f;
                RequestDecision();
            }
            else
            {
                m_currentTime += Time.fixedDeltaTime;
            }
        }
    }
}