import dgeb

if __name__ == "__main__":
    model = dgeb.get_model("facebook/esm2_t6_8M_UR50D", layers="mid")
    tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
    tasks = [task for task in tasks if "retrieval" in task.metadata.type] # filter for retrieval tasks

    # run the evaluation
    evaluation = dgeb.DGEB(tasks=tasks)
    evaluation.run(model)