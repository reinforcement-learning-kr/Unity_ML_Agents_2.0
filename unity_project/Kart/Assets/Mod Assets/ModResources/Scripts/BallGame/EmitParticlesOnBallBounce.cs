using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(ParticleSystem))]
public class EmitParticlesOnBallBounce : MonoBehaviour {
    ParticleSystem pSystem;

    public bool emitOnCollision = true;
    public bool emitOnKick = true;

#if UNITY_TEMPLATE_BALLGAME

    private void Awake() {
        pSystem = GetComponent<ParticleSystem>();
        if (emitOnCollision) {
            TeamBallGame.Gameplay.BallBounce.OnExecute += BallBounce_OnExecute;
            void BallBounce_OnExecute(TeamBallGame.Gameplay.BallBounce obj) {
                if (obj.collision.impulse.sqrMagnitude > 2f) {
                    pSystem.Play();
                }
            }
        }
        if (emitOnKick) {
            TeamBallGame.Gameplay.BallIsLaunched.OnExecute += BallIsLaunched_OnExecute;
            void BallIsLaunched_OnExecute(TeamBallGame.Gameplay.BallIsLaunched obj) {
                pSystem.Play();
            }
        }


    }

#endif


}
