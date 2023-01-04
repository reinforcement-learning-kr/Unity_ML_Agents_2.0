using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System;
using System.Collections;
using System.Collections.Generic;

public class HardExplorationAgent : Agent
{
    Rigidbody mAgentRb;

    public GameObject key1;
    public GameObject key2;
    public GameObject key3;

    public GameObject gate1;
    public GameObject gate2;

    public float speed = 1f;
    public float rotationSpeed = 150f;
    public float jumpForce = 7.5f;

    public bool canJump = false;

    public GameObject walls;

    private List<GameObject> room1_walls = new List<GameObject>();
    private List<GameObject> room2_walls = new List<GameObject>();

    public override void Initialize()
    {
        mAgentRb = GetComponent<Rigidbody>();

        room1_walls.Add(walls.transform.GetChild(0).gameObject);
        room1_walls.Add(walls.transform.GetChild(1).gameObject);
        room1_walls.Add(walls.transform.GetChild(2).gameObject);
        room1_walls.Add(walls.transform.GetChild(3).gameObject);
        room1_walls.Add(walls.transform.GetChild(4).gameObject);

        room2_walls.Add(walls.transform.GetChild(2).gameObject);
        room2_walls.Add(walls.transform.GetChild(3).gameObject);
        room2_walls.Add(walls.transform.GetChild(5).gameObject);
        room2_walls.Add(walls.transform.GetChild(6).gameObject);
        room2_walls.Add(walls.transform.GetChild(7).gameObject);
        room2_walls.Add(walls.transform.GetChild(8).gameObject);
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(20, 0.5f, -7.5f);
        transform.localRotation = Quaternion.Euler(0, 0, 0);
        mAgentRb.velocity = Vector3.zero;
        mAgentRb.angularVelocity = Vector3.zero;

        key1.transform.localPosition = new Vector3(8.5f, 1, -8.5f);
        key2.transform.localPosition = new Vector3(7.91f, 3, 4.35f);
        key3.transform.localPosition = new Vector3(-4f, 1, -3f);

        key1.SetActive(true);
        key2.SetActive(true);
        key3.SetActive(true);

        gate1.SetActive(true);
        gate2.SetActive(true);

        // reset color
        foreach (GameObject wall in room1_walls)
        {
            wall.GetComponent<Renderer>().material.color = Color.white;
        }

        foreach (GameObject wall in room2_walls)
        {
            wall.GetComponent<Renderer>().material.color = Color.white;
        }
    }

    public void MoveAgent(ActionBuffers actionBuffers)
    {
        var discreteActions = actionBuffers.DiscreteActions;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        dirToGo = transform.forward * discreteActions[1];
        rotateDir = -transform.up * discreteActions[2];
        
        mAgentRb.transform.Translate(dirToGo * speed * Time.fixedDeltaTime, Space.World);
        
        transform.Rotate(rotateDir, Time.fixedDeltaTime * rotationSpeed);

        if (discreteActions[0] == 1 && canJump)
        {
            mAgentRb.AddForce(transform.up * jumpForce, ForceMode.VelocityChange);
            canJump = false;
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        MoveAgent(actions);
        
        if (transform.localPosition.y < -1f)
        {
            SetReward(-1f);
            EndEpisode();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("key1"))
        {
            key1.SetActive(false);
            gate1.SetActive(false);

            foreach (GameObject wall in room1_walls)
            {
                wall.GetComponent<Renderer>().material.color = new Color(255, 0, 0);
            }
        }
        else if (other.gameObject.CompareTag("key2"))
        {
            key2.SetActive(false);
            gate2.SetActive(false);

            foreach (GameObject wall in room2_walls)
            {
                wall.GetComponent<Renderer>().material.color = new Color(255, 150, 0);
            }
        }
        else if (other.gameObject.CompareTag("key3"))
        {
            SetReward(1f);
            EndEpisode();
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
            discreteActionsOut[2] = -1;
        }
    }
}
