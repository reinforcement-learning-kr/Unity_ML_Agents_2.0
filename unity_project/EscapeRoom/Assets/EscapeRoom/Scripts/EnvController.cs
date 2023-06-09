using System;
using UnityEngine;
using Unity.MLAgents;
using System.Collections.Generic;

public class EnvController : MonoBehaviour
{
    [Serializable]
    public class PlayerInfo
    {
        public BlockAgent Agent;
        [HideInInspector] public Vector3 StartingPos;
        [HideInInspector] public Quaternion StartingRot;
        [HideInInspector] public Rigidbody RbAgent;
    }

    public List<PlayerInfo> AgentList = new List<PlayerInfo>();
    public int MaxEnvironmentSteps = 25000;
    public GameObject Ground = null;
    public GameObject Door = null;
    public Transform TrapTr = null;

    private int resetTimer;
    private Bounds areaBounds;
    private Dictionary<BlockAgent, PlayerInfo> playerDict = new Dictionary<BlockAgent, PlayerInfo>();
    private SimpleMultiAgentGroup agentGroup;
    private int numberOfRemainPlayers;
    private float spawnAreaMarginMultiplier = 0.9f;

    private void Start()
    {

        areaBounds = Ground.GetComponent<Collider>().bounds;
        agentGroup = new SimpleMultiAgentGroup();
        foreach (var block in AgentList)
        {
            block.StartingPos = block.Agent.transform.position;
            block.StartingRot = block.Agent.transform.rotation;
            block.RbAgent = block.Agent.GetComponent<Rigidbody>();
            agentGroup.RegisterAgent(block.Agent);
        }
        numberOfRemainPlayers = AgentList.Count;

        ResetScene();
    }

    private void FixedUpdate()
    {
        resetTimer += 1;
        if (resetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            agentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }
    }

    public void GoalReached()
    {
        agentGroup.AddGroupReward(1f);
        Debug.Log("Goal arrived");
        agentGroup.EndGroupEpisode();
        ResetScene();
    }

    public void OpenDoor()
    {
        Door.gameObject.SetActive(false);
    }

    public void CloseDoor()
    {
        Door.gameObject.SetActive(true);
    }

    public void KilledByTrap(BlockAgent agent)
    {
        numberOfRemainPlayers--;
        if (numberOfRemainPlayers == 0)
        {
            Debug.Log("End game...");
            agentGroup.EndGroupEpisode();
            ResetScene();
        }
        else
        {
            Debug.Log($"Remain players : {numberOfRemainPlayers}");
            agent.gameObject.SetActive(false);
            OpenDoor();
        }
    }

    private void ResetScene()
    {
        TrapTr.position = GetRandomSpawnPos(yPos: 0.01f, extensionSize: 6f);

        foreach (var agent in AgentList)
        {
            var pos = GetRandomSpawnPos(yPos: 0.5f, extensionSize: 1.5f);
            var rot = GetRandomRot();

            agent.Agent.transform.SetPositionAndRotation(pos, rot);
            agent.RbAgent.velocity = Vector3.zero;
            agent.RbAgent.angularVelocity = Vector3.zero;
            agent.Agent.gameObject.SetActive(true);

            agentGroup.RegisterAgent(agent.Agent);
        }
        resetTimer = 0;

        numberOfRemainPlayers = AgentList.Count;
        CloseDoor();
    }

    private Vector3 GetRandomSpawnPos(float yPos, float extensionSize)
    {
        var foundNewSpawnLocation = false;
        var randomSpawnPos = Vector3.zero;

        while (foundNewSpawnLocation == false)
        {
            var randomPosX = UnityEngine.Random.Range(-areaBounds.extents.x * spawnAreaMarginMultiplier,
                areaBounds.extents.x * spawnAreaMarginMultiplier);

            var randomPosZ = UnityEngine.Random.Range(-areaBounds.extents.z * spawnAreaMarginMultiplier,
                areaBounds.extents.z * spawnAreaMarginMultiplier);
            randomSpawnPos = Ground.transform.position + new Vector3(randomPosX, 0.5f, randomPosZ);

            if (Physics.CheckBox(randomSpawnPos, new Vector3(extensionSize, 0.01f, extensionSize)) == false)
            {
                foundNewSpawnLocation = true;
            }
        }
        randomSpawnPos.y = yPos;
        return randomSpawnPos;
    }

    private Quaternion GetRandomRot()
    {
        return Quaternion.Euler(0, UnityEngine.Random.Range(0.0f, 360.0f), 0);
    }

}
