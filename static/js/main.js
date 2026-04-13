// Initialize an empty network on load
const container = document.getElementById('network');
let network = new vis.Network(container, {nodes: [], edges: []}, {});

async function askQuestion() {
    const query = document.getElementById('queryInput').value;
    if (!query) return;

    // UI Loading State
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('loadingSpinner');
    const askBtn = document.getElementById('askBtn');
    const answerBox = document.getElementById('answerBox');

    btnText.style.display = 'none';
    spinner.style.display = 'block';
    askBtn.disabled = true;
    answerBox.style.display = 'none';

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        // Display Text Answer
        answerBox.innerHTML = `<strong style="color: var(--irminsul-green); font-family: var(--font-heading);">Knowledge Restored:</strong> <br><br> ${data.answer.replace(/\n/g, '<br>')}`;
        answerBox.style.display = 'block';

        // Vis.js Irminsul Theme Configuration
        const options = {
            nodes: {
                shape: 'dot',
                size: 25,
                font: { 
                    size: 16, 
                    color: '#e2e8f0',
                    face: 'Cinzel, serif',
                    vadjust: 5
                },
                color: { 
                    background: '#0d222b', // Dark teal inner
                    border: '#38ef7d',     // Bright green outer
                    highlight: { background: '#11998e', border: '#ffffff' },
                    hover: { background: '#11998e', border: '#ffffff' }
                },
                borderWidth: 2,
                shadow: { enabled: true, color: '#38ef7d', size: 15, x: 0, y: 0 }
            },
            edges: {
                font: { 
                    size: 12, 
                    align: 'middle', 
                    color: '#a8ffcc', // Mint green text
                    strokeWidth: 2,
                    strokeColor: '#0a0a12' // Abyssal background to make text readable
                },
                color: { 
                    color: '#11998e',
                    highlight: '#38ef7d',
                    hover: '#38ef7d'
                },
                width: 2,
                smooth: { type: 'continuous' }
            },
            physics: { 
                barnesHut: { 
                    gravitationalConstant: -4000, 
                    springLength: 200,
                    springConstant: 0.04
                } 
            },
            interaction: {
                hover: true,
                tooltipDelay: 200
            }
        };

        // Render the Graph
        network.setData({
            nodes: new vis.DataSet(data.nodes),
            edges: new vis.DataSet(data.edges)
        });
        network.setOptions(options);

    } catch (err) {
        alert("The Ley Lines are disrupted (Cannot connect to backend).");
        console.error(err);
    } finally {
        // Reset UI
        spinner.style.display = 'none';
        btnText.style.display = 'block';
        askBtn.disabled = false;
    }
}

// Allow pressing "Enter" to submit
document.getElementById("queryInput").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("askBtn").click();
    }
});