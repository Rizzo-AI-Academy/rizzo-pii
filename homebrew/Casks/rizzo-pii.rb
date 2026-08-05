cask "rizzo-pii" do
  version "1.0.0"
  sha256 "dbc01e6a1890ee114449fc07df12e2150ef5813d81d62d0940c131f29c2db9b2"

  url "https://github.com/Rizzo-AI-Academy/rizzo-pii/releases/download/v#{version}/Rizzo-PII-#{version}-macOS-arm64.dmg"
  name "Rizzo PII"
  desc "Local, reversible PII anonymization for Italian legal text"
  homepage "https://rizzo-ai-academy.github.io/rizzo-pii/"

  livecheck do
    url "https://github.com/Rizzo-AI-Academy/rizzo-pii/releases"
    strategy :github_latest
  end

  depends_on :macos

  # Apple Silicon only: the released DMG is arm64 (see build_mac.sh). When an
  # x86_64 build is published, add an `on_intel` URL branch here.
  app "Rizzo PII.app"
end
