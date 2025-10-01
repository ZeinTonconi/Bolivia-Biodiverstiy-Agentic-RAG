import { useState, useRef, useEffect } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Container,
} from "@mui/material";

export default function ChatUI() {
  const API_URL = import.meta.env.VITE_API_URL || "/ask";
  const [messages, setMessages] = useState([
    { id: 1, role: "system", text: "Preguntame sobre la biodiversidad de Bolivia" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function addMessage(role, text, sources) {
    setMessages(prev => [...prev, { id: Date.now(), role, text, sources }]);
  }

  async function send() {
    const q = input.trim();
    if (!q) return;
    addMessage("user", q);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q })
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Server error: ${res.status} ${txt}`);
      }
      const data = await res.json();
      const answer = data.answer ?? data.message ?? "";
      const sources = data.sources ?? null;
      addMessage("assistant", answer, sources);
    } catch (err) {
      addMessage("assistant", `Existio un error, intentelo de nuevo mas tarde`);
    } finally {
      setLoading(false);
    }
  }

  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading) send();
    }
  }

  return (
    <Box sx={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Agentic RAG — Chat
          </Typography>
          {loading ? <CircularProgress color="inherit" size={20} /> : <Typography variant="body2">Ready</Typography>}
        </Toolbar>
      </AppBar>

      <Container sx={{ flex: 1, py: 2, overflow: "auto" }}>
        <List>
          {messages.map(msg => (
            <ListItem key={msg.id} sx={{ justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
              <Paper
                elevation={2}
                sx={{
                  p: 2,
                  maxWidth: "70%",
                  bgcolor: msg.role === "user" ? "primary.main" : "grey.100",
                  color: msg.role === "user" ? "white" : "black"
                }}
              >
                <ListItemText
                  primary={
                    <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
                      {msg.text}
                    </Typography>
                  }
                />
                {msg.sources && msg.sources.length > 0 && (
                  <Box mt={1}>
                    <Typography variant="caption" fontWeight="bold">Sources:</Typography>
                    <ul style={{ margin: 0, paddingLeft: "1rem" }}>
                      {msg.sources.map((s, i) => (
                        <li key={i}>
                          <a href={s} target="_blank" rel="noreferrer">{s}</a>
                        </li>
                      ))}
                    </ul>
                  </Box>
                )}
              </Paper>
            </ListItem>
          ))}

          {loading && (
            <ListItem>
              <Paper elevation={2} sx={{ p: 2, bgcolor: "grey.100" }}>
                <CircularProgress size={24} />
              </Paper>
            </ListItem>
          )}
          <div ref={bottomRef} />
        </List>
      </Container>

      <Box component="footer" sx={{ p: 2, borderTop: "1px solid #ddd" }}>
        <Box sx={{ display: "flex", gap: 2 }}>
          <TextField
            fullWidth
            multiline
            maxRows={3}
            placeholder="Pregunta sobre la biodiversidad en Bolivia"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={loading}
          />
          <Button
            variant="contained"
            onClick={send}
            disabled={loading || input.trim() === ""}
          >
            Send
          </Button>
        </Box>
      </Box>
    </Box>
  );
}
