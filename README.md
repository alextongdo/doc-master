# DocMaster
DocMaster is a unified platform designed for annotating PDF documents, model training, and inference, tailored for document question-answering. Importantly, as annotations, training, and inference occur on-device, it also safeguards privacy. The platform has been instrumental in driving several research prototypes concerning document analysis such as the AI assistant utilized by University of California San Diego’s (UCSD) International Services and Engagement Office (ISEO).

## Installation
The simplest method to deploy DocMaster is to clone this repo and use the provided Docker Compose file.

1. First, clone this repo.  
`git clone git@github.com:alextongdo/doc-master.git`

2. Install Docker.  
`https://docs.docker.com/get-docker/`

4. Navigate to where you have cloned the Github repo and use the Docker Hub Compose file.  
`docker compose up`

### Platform Incompatibility
If you encounter any platform errors, a Compose file is provided that builds the images from scratch using this Github repository, which is a more stable method of installation for x86 platforms.

1. First, follow steps 1 and 2 from above.

2. Navigate to where you have cloned the Github repo and use the non Docker Hub Compose file.  
`docker compose -f docker-compose-no-hub.yml up`