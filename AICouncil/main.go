package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	git "github.com/go-git/go-git/v5"
)

// OllamaAPIRequest represents the payload to send to the Ollama API
type OllamaAPIRequest struct {
	Code string `json:"code"`
}

// OllamaAPIResponse represents the response from the Ollama API
type OllamaAPIResponse struct {
	Comment string `json:"comment"`
	Motion  string `json:"motion"`
}

// PullRequestPayload represents the payload for GitHub PR
type PullRequestPayload struct {
	Title string `json:"title"`
	Body  string `json:"body"`
	Head  string `json:"head"`
	Base  string `json:"base"`
}

func main() {
	repoURL := "https://github.com/yourusername/your-repo.git"
	repoDir := "temp-repo"
	ollamaAPIURL := "https://ollama.yourdomain.com/comment"
	githubAPIURL := "https://api.github.com/repos/yourusername/your-repo/pulls"
	githubToken := os.Getenv("GITHUB_TOKEN") // Set your GitHub token in the environment

	// Clone the repository
	err := cloneRepo(repoURL, repoDir)
	if err != nil {
		fmt.Printf("Error cloning repo: %v\n", err)
		return
	}
	defer os.RemoveAll(repoDir)

	// Read code files
	code, err := readFiles(repoDir)
	if err != nil {
		fmt.Printf("Error reading files: %v\n", err)
		return
	}

	// Send to Ollama API
	comment, motion, err := sendToOllamaAPI(ollamaAPIURL, code)
	if err != nil {
		fmt.Printf("Error communicating with Ollama API: %v\n", err)
		return
	}

	fmt.Printf("Comment: %s\n", comment)
	fmt.Printf("Motion: %s\n", motion)

	// Facilitate dialogue (simplified simulation)
	if motion == "pass" {
		fmt.Println("Motion passed. Creating pull request...")
		err = createPullRequest(githubAPIURL, githubToken, "Add updates based on LLM motion", comment, "feature-branch", "main")
		if err != nil {
			fmt.Printf("Error creating pull request: %v\n", err)
		} else {
			fmt.Println("Pull request created successfully.")
		}
	} else {
		fmt.Println("Motion did not pass.")
	}
}

// cloneRepo clones a GitHub repository
func cloneRepo(repoURL, repoDir string) error {
	_, err := git.PlainClone(repoDir, false, &git.CloneOptions{
		URL: repoURL,
	})
	return err
}

// readFiles reads all code files from the repository directory
func readFiles(dir string) (string, error) {
	var codeBuffer bytes.Buffer

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && (filepath.Ext(path) == ".go" || filepath.Ext(path) == ".py" || filepath.Ext(path) == ".js") {
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer file.Close()

			_, err = io.Copy(&codeBuffer, file)
			if err != nil {
				return err
			}
		}
		return nil
	})

	return codeBuffer.String(), err
}

// sendToOllamaAPI sends code to the Ollama API for comments
func sendToOllamaAPI(apiURL, code string) (string, string, error) {
	requestBody := OllamaAPIRequest{Code: code}
	body, err := json.Marshal(requestBody)
	if err != nil {
		return "", "", err
	}

	resp, err := http.Post(apiURL, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	var response OllamaAPIResponse
	err = json.NewDecoder(resp.Body).Decode(&response)
	if err != nil {
		return "", "", err
	}

	return response.Comment, response.Motion, nil
}

// createPullRequest creates a pull request on GitHub
func createPullRequest(apiURL, token, title, body, head, base string) error {
	payload := PullRequestPayload{
		Title: title,
		Body:  body,
		Head:  head,
		Base:  base,
	}
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "token "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to create pull request: %s", string(body))
	}

	return nil
}
