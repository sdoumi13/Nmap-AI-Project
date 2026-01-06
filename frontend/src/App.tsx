import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import RouterPage from './pages/RouterPage'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/router" element={<RouterPage />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App

