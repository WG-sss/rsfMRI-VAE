import boto3
import os

def list_subject_ids(bucket_name, prefix):
    """
    List all subject IDs in the HCP_1200 directory.

    :param bucket_name: The name of the S3 bucket.
    :param prefix: The prefix of the directory in the S3 bucket.
    :return: A list of subject IDs.
    """
    s3 = boto3.client('s3', aws_access_key_id='AKIAXO65CT57B6MJMPRN',
                      aws_secret_access_key='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')
    subject_ids = []

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        for obj in response.get('CommonPrefixes', []):
            subject_id = obj['Prefix'].strip('/').split('/')[-1]
            subject_ids.append(subject_id)
    except Exception as e:
        print(f"An error occurred while listing subject IDs: {e}")

    return subject_ids

def download_hcp_rfMRI_data(bucket_name, path_prefix, subject_id, local_base_path='~/Datasets/HCP_1200'):
    """
    Download specific rfMRI REST files from the HCP dataset for a given subject.

    :param bucket_name: The name of the S3 bucket.
    :param subject_id: The ID of the subject.
    :param local_base_path: The base local path where the files will be saved.
    """
    prefix = f'HCP_1200/{subject_id}' + path_prefix
    file_pattern = "Atlas_MSMAll_hp2000_clean"
    file_suffix = ".dtseries.nii"
    s3 = boto3.client('s3', aws_access_key_id='AKIAXO65CT57B6MJMPRN',
                      aws_secret_access_key='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')

    # Create local directory for the subject if it doesn't exist
    subject_local_path = os.path.join(local_base_path, subject_id)
    if not os.path.exists(subject_local_path):
        print("local directory not found, creating.")
        os.makedirs(subject_local_path)

    # List all objects within the specified subject's directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            file_name = obj['Key']
            if "7T" in file_name:
                continue
                print(f"skipped {file_name}")
            elif file_pattern in file_name and file_name.endswith(file_suffix):
                subject_local_file = os.path.join(subject_local_path, subject_id + '_' + os.path.basename(file_name))
                if not os.path.exists(subject_local_file):
                    print(f"Trying download {file_name}")
                    try:
                        s3.download_file(bucket_name, file_name, subject_local_file)
                        print(f"Downloaded {file_name} to {subject_local_path}")
                    except Exception as e:
                        print(f"An error occurred while downloading files for subject {subject_id}: {e}")
                else:
                    print(f"{file_name} has already downloaded")
    else:
        print(f"No files found for subject {subject_id} in the HCP dataset.")


if __name__ == "__main__":
    bucket_name = 'hcp-openaccess'
    hcp_prefix = 'HCP_1200/'
    subject_ids = list_subject_ids(bucket_name, hcp_prefix)

    path_prefix = ['/MNINonLinear/Results/rfMRI_REST1_LR/',
                   '/MNINonLinear/Results/rfMRI_REST1_RL/',
                   '/MNINonLinear/Results/rfMRI_REST2_LR/',
                   '/MNINonLinear/Results/rfMRI_REST2_RL/']

    for subject_id in subject_ids[:100]:
        for path in path_prefix:
            download_hcp_rfMRI_data(bucket_name, path, subject_id)

