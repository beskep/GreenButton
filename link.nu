# 작업폴더 링크 생성
def main [] {
  let workspace = open 'config/.config.toml'
    | get workspace
    | path expand

  pwsh -Command $"New-Item -ItemType SymbolicLink -Path work -Target ($workspace)"
}
