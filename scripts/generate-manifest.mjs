import { promises as fs } from "fs"
import path from "path"

const root = process.cwd()
const publicDir = path.join(root, "public", "data")

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true })
}

async function main() {
  await ensureDir(publicDir)

  const entries = await fs.readdir(publicDir)
  const jsonFiles = entries.filter((name) => name.endsWith(".json") && name !== "manifest.json")

  const manifest = {
    files: jsonFiles.sort(),
  }
  await fs.writeFile(path.join(publicDir, "manifest.json"), JSON.stringify(manifest, null, 2))
}

main().catch((error) => {
  console.error("Failed to generate manifest:", error)
  process.exit(1)
})
