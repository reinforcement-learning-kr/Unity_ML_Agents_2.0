using System.Linq;
using Unity.Play.Publisher.Editor;
using UnityEngine;

namespace Unity.Tutorials
{
    /// <summary>
    /// Contaisn all the callbacks needed for the Build And Publish tutorial
    /// </summary>
    [CreateAssetMenu(fileName = "PublishCriteria", menuName = "Tutorials/Microgame/PublishCriteria")]
    class PublishCriteria : ScriptableObject
    {
        static PublisherWindow publisherWindow;
        public bool IsNotDisplayingFirstTimeInstructions()
        {
            if (!IsWebGLPublisherOpen()) { return false; }
            return (!string.IsNullOrEmpty(publisherWindow.CurrentTab) && publisherWindow.CurrentTab != PublisherWindow.TabIntroduction);
        }

        public bool IsUserLoggedIn()
        {
            if (!IsWebGLPublisherOpen()) { return false; }
            return (publisherWindow.CurrentTab != PublisherWindow.TabNotLoggedIn);
        }

        public bool IsBuildBeingUploaded()
        {
            if (!IsWebGLPublisherOpen()) { return false; }
            switch (PublisherUtils.GetCurrentPublisherState(publisherWindow))
            {
                case PublisherState.Upload:
                case PublisherState.Process:
                    return true;
                default: break;
            }
            return !string.IsNullOrEmpty(PublisherUtils.GetUrlOfLastPublishedBuild(publisherWindow));
        }

        public bool IsBuildPublished()
        {
            if (!IsWebGLPublisherOpen()) { return false; }
            return !string.IsNullOrEmpty(PublisherUtils.GetUrlOfLastPublishedBuild(publisherWindow));
        }

        public bool AtLeastOneBuildIsRegistered()
        {
            if (!IsWebGLPublisherOpen()) { return false; }
            switch (PublisherUtils.GetCurrentPublisherState(publisherWindow))
            {
                case PublisherState.Zip:
                case PublisherState.Upload:
                case PublisherState.Process:
                    return true;
                default: break;
            }
            int availableBuilds = PublisherUtils.GetAllBuildsDirectories()
                                            .Where(p => (p != string.Empty)
                                                        && PublisherUtils.BuildIsValid(p)).Count();
            return availableBuilds > 0;
        }

        bool IsWebGLPublisherOpen()
        {
            if (publisherWindow) { return true; }
            publisherWindow = PublisherWindow.FindInstance();
            return false;
        }
    }
}
