import supervisely as sly

remote_path = '/model-benchmark/41774_COCO-100 (det)/'

api = sly.Api.from_env()

# api.file.download_directory(sly.env.team_id(), remote_path, './41774_COCO-100 (det)')

project_id = 41774

sly.download_project(api, project_id, './data', save_image_info=True)