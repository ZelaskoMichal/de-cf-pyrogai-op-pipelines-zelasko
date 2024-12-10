#!/bin/bash -e

if [[ ! $USER_GITHUB_TOKEN ]]; then
    MSG='Secret `USER_GITHUB_TOKEN` was not found.'
    MSG="${MSG} It should be defined with scopes \`repo:*\` "
    MSG="${MSG} and \`workflow\` and \`Enable SSO\` should be done too."
    MSG="${MSG} Go here - https://github.com/settings/tokens - to create a Personal Access Token."
    echo "::error::${MSG}"
    exit 1
fi

curl_output=$(curl --silent --retry 3 --retry-delay 2 --retry-max-time 15 -sS -f -I -H "Authorization: token ${USER_GITHUB_TOKEN}" https://api.github.com | head -n 1 | grep -oE 'HTTP/([0-9]\.[0-9]+|[0-9]+) ([0-9]+)' | sed -E 's/HTTP\/([0-9]\.[0-9]+|[0-9]+) ([0-9]+)/\2/')
if ! [[ ${curl_output} == "200" ]]; then
    MSG="request to github rest api failed with status code ${curl_output}. check if the token value is set correctly"
    MSG="${MSG} in github secret 'USER_GITHUB_TOKEN' and if the token has sufficient permissions."
    echo "::error::${MSG}"
    exit 1
fi

scopes=$(curl --silent --retry 3 --retry-delay 2 --retry-max-time 15 -sS -f -I -H "Authorization: token ${USER_GITHUB_TOKEN}" https://api.github.com | grep ^x-oauth-scopes: | cut -d' ' -f2- | tr -d "[:space:]" | tr ',' '\n')
if [ $? -ne 0 ]; then
    echo "::error::failed to retrieve scope of USER_GITHUB_TOKEN."
    exit 1
fi

if ! [[ $scopes == *"repo"* ]] || ! [[ $scopes == *"workflow"* ]] || ! [[ $scopes == *"packages"* ]]; then
    MSG="USER_GITHUB_TOKEN should have 'repo + workflow + read:packages' scope. Please check if you have all three scopes defined for your token."
    MSG="${MSG} To update scope, open github profile -> go to 'Developer Settings' -> select 'personal access tokens' -> click on 'Tokens' -> "
    MSG="${MSG} click on token name you want to update -> select scopes -> click on 'update token' button."
    echo "::error::${MSG}"
    exit 1
fi

exp_date=$(curl --silent --retry 3 --retry-delay 2 --retry-max-time 15 -sS -f -I -H "Authorization: token ${USER_GITHUB_TOKEN}" https://api.github.com | grep ^github-authentication-token-expiration: | cut -d' ' -f2-)
if [ $? -ne 0 ]; then
    echo "::error::failed to retrieve expiration date of USER_GITHUB_TOKEN."
    exit 1
fi

if [ "$exp_date" ]; then
exp_timestamp=$(date -d "$exp_date" +%s)
current_timestamp=$(date +%s)
if [ "$exp_timestamp" -le "$current_timestamp" ]; then
    MSG='USER_GITHUB_TOKEN has been expired. kindly update this token.'
    echo "::error::${MSG}"
    exit 1
fi
fi

repos=$(curl --silent --retry 3 --retry-delay 2 --retry-max-time 15 -L -H "Accept: application/vnd.github+json" -H "Authorization: Bearer ${USER_GITHUB_TOKEN}" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/orgs/procter-gamble/repos)
count=0; for repo in "${repos[@]}"; do ((count++)); done
if [ $count -eq 0 ]; then
MSG="SSO is not configured for 'USER_GITHUB_TOKEN'. To configure the SSO, open github profile -> go to 'Developer Settings' -> select 'personal access tokens' -> click on 'Tokens' -> "
MSG="${MSG} click on 'Configure SSO' (present right of your token name) -> click on the organization name"
echo "::error::${MSG}"
exit 1
fi
echo "github token validation is successful. token has scopes: ${scopes} and expiration date is ${exp_date}"
