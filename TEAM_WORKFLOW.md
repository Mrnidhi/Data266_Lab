# Team Git workflow

Everyone uses **https://github.com/Mrnidhi/Data266_Lab**. Each person needs their own GitHub account added as a collaborator by the repository owner. Do not share credentials. Collaborator invitations have not been sent by this preparation.

## Member ownership

Create your own named folder under `task1_llm`, `task2_sentiment`, and `task3_gan`, following Srinidhi's folder structure. Each member owns their model source, notebook, configuration, processed data, outputs, checkpoint references and analysis. Do not reuse another member's implementation as independent work.

Coordinate edits to the root README, dependencies, utilities, common evaluation scripts and combined report. Record whose run produced each result. Agree on label mappings, fixed evaluation inputs and metric definitions before comparing scores.

## Contribution workflow

1. Clone the shared repository and pull the current `main` branch.
2. Create a branch such as `srinidhi/lab1` or `<name>/lab1`.
3. Change your own member folders and test the work.
4. Commit source, readable results, unedited logs and manifests. Keep large datasets, binaries, private paths and secrets out of ordinary Git.
5. Push your branch and open a pull request for teammate review before merging. Do not force-push shared main.

The initial preparation establishes the first commit. Subsequent work can use this branch/review workflow. RunPod and college machines should use the same repository, keeping run outputs and hardware disclosures distinct.

## Submission

Everyone contributes here, but the lab asks for **one team submission and one combined report** referencing each person's independent work. Pushing to GitHub does not submit to Canvas or Kaggle. Confirm which teammate will handle the final upload according to Canvas instructions.
