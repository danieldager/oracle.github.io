require('dotenv').config()
const { v4 } = require('uuid')

const express = require('express')
const app = express()
const port = 3000

const { MongoClient } = require('mongodb')
const uri = process.env.MONGO_URL
const client = new MongoClient(uri)
const db = client.db('Oracle')
const trials = db.collection('trials')

async function test() {
  const res = await db.command({ ping: 1 })
  console.log("connection:", res)
}
test()

// Allows for parsing of json
app.use(express.json())

// Serve static files from 'public'
app.use(express.static('public'))

// Endpoint for serving uuids 
app.get('/get', async (req, res) => { res.json({ uuid: v4() }) })

// Endpoint for routing data to the database
app.post('/post', async (req, res) => {
  try {
    // Unpack request body
    const { uuid, data } = req.body
    
    // Find trial in database
    var trial = await trials.findOne({ uuid: uuid })

    // If it does not, create it
    if (trial == null) {
      trial = {
        data: data,
        uuid: uuid,
        timestamp: Date.now(),
      }
      await trials.insertOne(trial) // Needs error handling
      res.json({ message: "successful insert" })
    }

    // If it does, concat old data with new data
    else { 
      const concat = [...trial.data, ...data ]
      await trials.updateOne({ uuid: uuid }, { $set: { data: concat } }) // Needs error handling
      res.json({ message: "successful update" })
    }
  }

  // Error handling
  catch (error) {
    res.status(500).send(error)
  }
})

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`)
})