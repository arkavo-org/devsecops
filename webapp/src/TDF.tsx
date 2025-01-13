import React, { useEffect, useState } from 'react';
import { useKeycloak } from '@react-keycloak/web';
import {AuthProvider, HttpRequest, NanoTDFDatasetClient} from '@opentdf/sdk';
import {DatasetConfig} from "@opentdf/sdk/nano";

const TDFContent: React.FC = () => {
  const { keycloak } = useKeycloak();
  const [count, setCount] = useState(0);
  const [tdfClient, setTdfClient] = useState<NanoTDFDatasetClient | null>(null);
  const [encryptedData, setEncryptedData] = useState<ArrayBuffer | null>(null);
  const [decryptedData, setDecryptedData] = useState<string | null>(null);
  const [status, setStatus] = useState<{
    type: 'info' | 'error' | 'success' | null;
    message: string | null;
  }>({ type: null, message: null });

  useEffect(() => {
    (async () => {
      if (!keycloak.authenticated) return;

      try {
        setStatus({ type: 'info', message: 'Initializing TDF client...' });

        const authProvider: AuthProvider = {
          async updateClientPublicKey(): Promise<void> {
            // nothing
            return
          },
          withCreds(httpReq: HttpRequest): Promise<HttpRequest> {
            httpReq.headers.Authorization = `Bearer ${keycloak.token}`;
            return Promise.resolve(httpReq);
          }
        }
        const client = new NanoTDFDatasetClient({
          authProvider,
          kasEndpoint: import.meta.env.VITE_KAS_ENDPOINT,
        } as DatasetConfig);

        setTdfClient(client);
        setStatus({ type: 'success', message: 'TDF client initialized successfully' });
      } catch (error) {
        console.error('TDF initialization failed:', error);
        setStatus({
          type: 'error',
          message: `Failed to initialize TDF client: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`,
        });
      }
    })();
  }, [keycloak.authenticated, keycloak.token]);

  const handleEncrypt = async () => {
    if (!tdfClient) return;
    try {
      setStatus({ type: 'info', message: 'Encrypting data...' });
      const data = `Count value: ${count}`;
      const encrypted = await tdfClient.encrypt(data);
      setEncryptedData(encrypted);
      setDecryptedData(null);
      setStatus({ type: 'success', message: 'Data encrypted successfully' });
    } catch (error) {
      console.error('Encryption failed:', error);
      setStatus({
        type: 'error',
        message: `Encryption failed: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
      });
    }
  };

  const handleDecrypt = async () => {
    if (!tdfClient || !encryptedData) return;
    try {
      setStatus({ type: 'info', message: 'Decrypting data...' });
      const decrypted = await tdfClient.decrypt(encryptedData);
      const decoder = new TextDecoder();
      const decryptedText = decoder.decode(decrypted);
      setDecryptedData(decryptedText);
      setStatus({ type: 'success', message: 'Data decrypted successfully' });
    } catch (error) {
      console.error('Decryption failed:', error);
      setStatus({
        type: 'error',
        message: `Decryption failed: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
      });
    }
  };

  const arrayBufferToHex = (buffer: ArrayBuffer): string => {
    return Array.from(new Uint8Array(buffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join(' ');
  };

  if (!keycloak.authenticated) {
    return (
      <div>
        <button onClick={() => keycloak.login()}>Log in</button>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="user-info">
        <p>Welcome {keycloak.tokenParsed?.preferred_username}</p>
        <button onClick={() => keycloak.logout()}>Log out</button>
      </div>
      <h1>Secure Vite + React App + OpenTDF + Keycloak</h1>
      {status.message && (
        <div
          className={`mt-4 p-4 rounded ${
            status.type === 'error'
              ? 'bg-red-100 text-red-700'
              : status.type === 'success'
                ? 'bg-green-100 text-green-700'
                : 'bg-blue-100 text-blue-700'
          }`}
        >
          {status.message}
        </div>
      )}
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <button onClick={handleEncrypt} disabled={!tdfClient} className="ml-4">
          Encrypt Count
        </button>
        <button
          onClick={handleDecrypt}
          disabled={!tdfClient || !encryptedData}
          className="ml-4"
        >
          Decrypt Data
        </button>
        {encryptedData && (
          <div className="mt-4">
            <h3>Encrypted Data</h3>
            <pre className="bg-gray-100 p-2 rounded font-mono text-sm">
              {arrayBufferToHex(encryptedData)}
            </pre>
          </div>
        )}
        {decryptedData && (
          <div className="mt-4">
            <h3>Decrypted Data</h3>
            <pre className="bg-gray-100 p-2 rounded font-mono text-sm">
              {decryptedData}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default TDFContent;
