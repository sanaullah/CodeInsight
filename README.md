<p align="center">
  <img src="assets/hero_banner.jpg" width="800" alt="CodeInsight Logo">
</p>

# CodeInsight

CodeInsight acts as an on-demand, intelligent sidekick designed to run alongside your development workflow. As you build applications using tools like Cursor, CodeInsight offers a proactive "second pair of eyes" to identify and help resolve issues that aren't immediately apparent during the coding phase.

By orchestrating a swarm of specialized AI agents, the platform looks beyond the surface of your code to uncover hidden complexities. It empowers you to build with greater confidence by automatically detecting security risks, performance bottlenecks, and architectural inconsistencies in real-time—allowing you to fix problems as they arise, rather than waiting for them to surface later.

## Supported Languages

CodeInsight supports analysis for a wide range of programming languages, including:

- **Core**: Python, JavaScript, TypeScript, Java, C/C++, C#
- **Systems**: Go, Rust, Swift, Kotlin
- **Web**: HTML, CSS, PHP, Ruby
- **Scripting**: Shell, PowerShell, Lua, Perl
- **Data/Config**: SQL, YAML, JSON, XML, Markdown, R, MATLAB
- **Other**: BoxLang, ColdFusion, Dart, Scala


## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker (for running Langfuse locally)

### Installation

You can set up CodeInsight using the provided automated scripts (recommended) or follow the manual steps.

#### **Option 1: Automated Setup (Recommended)**

The setup scripts automatically create a virtual environment, install dependencies, prepare configuration files, start Langfuse services (via Docker), and initialize the database.

1.  **Run Setup Script:**
    **On Windows:**
    - **Double-click** `setup.bat` in the project root.
    - *Or run via PowerShell:*
      ```powershell
      powershell.exe -ExecutionPolicy Bypass -File "setup.ps1"
      ```
    **On Linux/macOS:**
    ```bash
    python3 setup.py
    ```

2.  **Configure Keys:**
    Once the script finishes, it will have created a `.env` file for you. Open it and add your required keys:
    - `OPENAI_API_KEY`: Your model provider API key.
    - `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY`: (Optional) For observability.

---

#### **Option 2: Manual Setup (Fallback)**

If the automated scripts fail or if you prefer a custom setup, follow these steps:

1.  **Create & Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/macOS:
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Configuration Files**:
    Copy the example files:
    ```bash
    cp .env.example .env
    cp config.yaml.example config.yaml
    ```

4.  **Configure Keys**:
    Edit `.env` and add your required keys:
    - `OPENAI_API_KEY` (Required for LLM)
    - `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY` (Optional)

5.  **Start Langfuse (Docker)**:
    ```bash
    docker compose -f langfuse/docker-compose.yml up -d
    ```

6.  **Initialize Database**:
    ```bash
    python scripts/init_database.py
    ```

---

### Starting the Application

Once setup is complete, ensure your virtual environment is active and run:

```bash
streamlit run ui/app.py
```


## Acknowledgments

**Democratizing AI Development.** One of the biggest barriers to innovation is the cost of compute. CodeInsight exists today largely because of [Nano-GPT](https://nano-gpt.com), a platform that provides access to high-performance open-source models at a fraction of the usual cost.

We believe powerful AI tools should be accessible to everyone—students, hobbyists, and startups alike—without breaking the bank. If you're looking to power your own projects with affordable, high-quality API endpoints, you can get started here:

👉 **[Get Access to Nano-GPT](https://nano-gpt.com/invite/ck3mnXrF)**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
