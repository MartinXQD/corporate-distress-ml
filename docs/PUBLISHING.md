# Public release policy

Only this new extension may be initialized/pushed as its own Git repository. Verify the Git root, authenticated owner and proposed repository name first. The source workspace is not staged. No force push, parent remote change or unknown-repository replacement is allowed.

Public allowlist: new source, tests with synthetic fixtures, configurations, aggregate results and figures, sanitized notebooks, the two research/teaching reports and their editable sources, attribution, citation and reproduction instructions. Exclusions: source/processed/SMOTE data, complete row predictions, model artifacts, original team submissions, chats, feedback, credentials, environment files, virtual environments, font files, caches, local source manifests and draft resume bullets.

Scan both the working-tree allowlist and Git index for secrets, private absolute paths, excluded file types/names and oversize files. Extract PDF text and inspect notebook outputs as part of that check. Verify all relative README links. New-code license is pending; do not attach an invented open-source license to this work, the source data or team materials.

After a successful push, verify remote visibility, owner, branch commit and key file presence. Save real evidence. A local commit, staged package or upload UI is not a publication certificate. If authentication or repository creation fails, deliver local outputs and identify the exact missing step without inventing a URL or repeatedly requesting tokens.
