#!/bin/bash -e

# Robust add-remove-commit-push command for GitHub Actions. We must make sure
# the current process always wins when we have concurrent pushes.

CMD=$1        # add|rm
FILE=$2       # the file to add or remove
BRANCH=$3     # branch
FULL_REPO=$4  # https://whatever.com/org/repo, or git@something.com:org/repo

RETRY=10
PUSH_OK=

for ((I=0; I<RETRY; I++)); do
    ERR=

    # Add/remove and commit
    if [[ $CMD == add ]]; then
        git add "$FILE"
        git commit -m "Add $FILE" --allow-empty
    elif [[ $CMD == rm ]]; then
        git rm -f "$FILE" || break  # if file does not exist, just exit
        git commit -m "Remove $FILE"
    fi

    # Attempt to push
    git push "$FULL_REPO" HEAD:"$BRANCH" || ERR=1

    if [[ ! $ERR ]]; then
        # Success
        break
    fi

    echo "Push not successful: updating branch and retrying immediately"
    if [[ -d "$FILE" ]]; then
        SLASH='/'
    else
        SLASH=''
    fi
    if [[ $CMD == add ]]; then
        BAK=$(mktemp)
        rsync -av "${FILE}${SLASH}" "${BAK}${SLASH}"
    fi
    git fetch origin "$BRANCH"
    git reset --hard origin/"$BRANCH"
    if [[ $CMD == add ]]; then
        rsync -av "${BAK}${SLASH}" "${FILE}${SLASH}"
        rm -rf "${BAK}"
    fi
done

if [[ $ERR ]]; then
    echo "Giving up after $RETRY attempts"
    exit 1
fi

echo "Push successful"
exit 0
