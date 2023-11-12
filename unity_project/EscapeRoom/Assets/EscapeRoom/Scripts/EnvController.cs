using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class EnvController : MonoBehaviour
{
    [Serializable]
    public class PlayerInfo
    {
        public BlockAgent Agent;
        [HideInInspector] public Vector3 StartingPos;
        [HideInInspector] public Rigidbody RbAgent;
    }

    public List<PlayerInfo> AgentList = new List<PlayerInfo>();
    public int MaxEnvironmentSteps = 2000;
    public GameObject Ground = null;
    public GameObject Door = null;
    public Transform TrapTr = null;

    private int trap_dir = 1;
    private int resetTimer;
    private Bounds areaBounds;

    private SimpleMultiAgentGroup agentGroup;
    public int numberOfRemainPlayers;
    private float spawnAreaMarginMultiplier = 0.8f;

    void Start()
    {
        areaBounds = Ground.GetComponent<Collider>().bounds;
        agentGroup = new SimpleMultiAgentGroup();
        foreach (var block in AgentList)
        {
            block.StartingPos = block.Agent.transform.position;
            block.RbAgent = block.Agent.GetComponent<Rigidbody>();
            agentGroup.RegisterAgent(block.Agent);
        }
        numberOfRemainPlayers = AgentList.Count;

        ResetScene();
    }

    void FixedUpdate()
    {
        resetTimer += 1;
        if (resetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            agentGroup.GroupEpisodeInterrupted();
            ResetScene(); 
        }

        MoveTrap();
    }

    public void GoalReached()
    {
        agentGroup.AddGroupReward(1f);
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
            agentGroup.EndGroupEpisode();
            ResetScene();
        }
        else
        {
            agent.gameObject.SetActive(false);
            OpenDoor(); 
        }
    }

    public void MoveTrap()
    {
        if (trap_dir == 1 && TrapTr.position.x >= 12)
        {
            trap_dir = -1;
        }

        if (trap_dir == -1 && TrapTr.position.x <= -12)
        {
            trap_dir = 1;
        }

        TrapTr.position = new Vector3(TrapTr.position.x + trap_dir*0.1f, TrapTr.position.y, TrapTr.position.z); 
    }

    private List<Vector2> GetRandomSpawnPos()
    {
        List<Vector2> randPosList = new List<Vector2>(); ;
        for (int i = 0; i < 4; ++i)
        {
            Vector2 randPos = new Vector2();
            while (true)
            {
                randPos = new Vector2(Ground.transform.position.x, Ground.transform.position.z) + new Vector2(
                    UnityEngine.Random.Range(-areaBounds.extents.x * spawnAreaMarginMultiplier, areaBounds.extents.x * spawnAreaMarginMultiplier),
                    UnityEngine.Random.Range(-areaBounds.extents.z * spawnAreaMarginMultiplier, areaBounds.extents.z * spawnAreaMarginMultiplier));

                bool again = false;
                foreach (Vector2 tmpPos in randPosList)
                {
                    if (Vector2.Distance(tmpPos, randPos) <= 5.0)
                    {
                        again = true;
                        break;
                    }
                }

                if (!again) { break; }
            }
            randPosList.Add(randPos);
        }
        return randPosList;
    }

    private void ResetScene()
    {
        List<Vector2> randPosList = GetRandomSpawnPos();

        TrapTr.position = new Vector3(randPosList[0].x, 0.01f, randPosList[0].y);

        int index = 1;
        foreach (var agent in AgentList)
        {
            var pos = new Vector3(randPosList[index].x, 0.5f, randPosList[index].y);

            agent.Agent.transform.position = pos;

            agent.RbAgent.velocity = Vector3.zero;
            agent.RbAgent.angularVelocity = Vector3.zero;
            agent.Agent.gameObject.SetActive(true);

            agentGroup.RegisterAgent(agent.Agent);
            index++;
        }
        resetTimer = 0;

        numberOfRemainPlayers = AgentList.Count;
        CloseDoor();
    }
}