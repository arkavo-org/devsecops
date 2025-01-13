// useBlueskyIntegration.ts
import { useEffect } from 'react';
import { useKeycloak } from '@react-keycloak/web';
import { BskyAgent } from '@atproto/api';

const BACKEND_API = import.meta.env.VITE_BACKEND_API_URL;
const BLUESKY_API_URL = 'https://bsky.social/xrpc';

interface BlueskyCredentials {
  did: string;
  password: string;
}

export const useBlueskyIntegration = () => {
  const { keycloak, initialized } = useKeycloak();
  const bskyAgent = new BskyAgent({ serviceURL: BLUESKY_API_URL });

  const checkBlueskyCredentials = async (): Promise<BlueskyCredentials | null> => {
    try {
      const response = await fetch(`${BACKEND_API}/user/bluesky-credentials`, {
        headers: {
          'Authorization': `Bearer ${keycloak.token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch Bluesky credentials');
      }
      
      const data = await response.json();
      return data.did ? data : null;
    } catch (error) {
      console.error('Error checking Bluesky credentials:', error);
      return null;
    }
  };

  const verifyBlueskyCredentials = async (credentials: BlueskyCredentials): Promise<boolean> => {
    try {
      await bskyAgent.login({
        identifier: credentials.did,
        password: credentials.password
      });
      await bskyAgent.logout();
      return true;
    } catch (error) {
      console.error('Error verifying Bluesky credentials:', error);
      return false;
    }
  };

  const generateSecurePassword = (): string => {
    const length = 16;
    const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
    return Array.from(crypto.getRandomValues(new Uint8Array(length)))
      .map(x => charset[x % charset.length])
      .join('');
  };

  const createBlueskyAccount = async () => {
    try {
      const userProfile = await keycloak.loadUserProfile();
      const email = userProfile.email;
      if (!email) throw new Error('No email found in user profile');

      const password = generateSecurePassword();
      const handle = `${email.split('@')[0]}.bsky.social`;

      await bskyAgent.createAccount({
        email,
        handle,
        password
      });

      const did = bskyAgent.session?.did;
      if (!did) throw new Error('No DID returned from account creation');

      await fetch(`${BACKEND_API}/user/bluesky-credentials`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${keycloak.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ did, password })
      });
    } catch (error) {
      console.error('Error creating Bluesky account:', error);
      throw error;
    }
  };

  useEffect(() => {
    const handleAuth = async () => {
      if (initialized && keycloak.authenticated) {
        try {
          const credentials = await checkBlueskyCredentials();
          
          if (!credentials) {
            await createBlueskyAccount();
          } else {
            const isValid = await verifyBlueskyCredentials(credentials);
            if (!isValid) {
              await createBlueskyAccount();
            }
          }
        } catch (error) {
          console.error('Error in Bluesky integration:', error);
        }
      }
    };

    handleAuth();

    // Listen for Keycloak events
    const handleOnAuthSuccess = () => {
      handleAuth();
    };

    keycloak.onAuthSuccess = handleOnAuthSuccess;

    return () => {
      keycloak.onAuthSuccess = undefined;
    };
  }, [initialized, keycloak.authenticated]);
};

// Add this to your App.tsx:
const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(localStorage.getItem('darkMode') === 'true');
  
  // Add the Bluesky integration hook
  useBlueskyIntegration();

  // Rest of your existing App code...
  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
    localStorage.setItem('darkMode', darkMode.toString());

    const applyDarkModeToFeedItems = () => {
      const feedItems = document.querySelectorAll('.feed-item');
      feedItems.forEach((item) => {
        item.classList.toggle('dark-mode', darkMode);
      });
    };

    const observer = new MutationObserver(applyDarkModeToFeedItems);
    observer.observe(document.body, { childList: true, subtree: true });

    applyDarkModeToFeedItems();

    return () => observer.disconnect();
  }, [darkMode]);

  return (
    <ReactKeycloakProvider authClient={keycloak}>
      <Router future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}>
        <AppRoutes darkMode={darkMode} onToggleDarkMode={() => setDarkMode(!darkMode)} />
      </Router>
    </ReactKeycloakProvider>
  );
};