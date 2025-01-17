from typing import Optional, Dict, Any
from boto3 import client
from botocore.exceptions import ClientError
from langchain.tools import Tool, StructuredTool
from typing import Optional


def create_cloudformation_tools() -> list[Tool]:
    """Create and return a list of CloudFormation-related tools."""

    def list_stacks() -> str:
        """List all CloudFormation stacks in the account."""
        try:
            cf_client = client('cloudformation')
            response = cf_client.list_stacks(
                StackStatusFilter=['CREATE_COMPLETE', 'UPDATE_COMPLETE', 'UPDATE_ROLLBACK_COMPLETE']
            )

            if not response.get('StackSummaries'):
                return "No active CloudFormation stacks found."

            stack_info = []
            for stack in response['StackSummaries']:
                stack_info.append(
                    f"Stack: {stack['StackName']}\n"
                    f"Status: {stack['StackStatus']}\n"
                    f"Created: {stack['CreationTime']}\n"
                )

            return "\n".join(stack_info)
        except ClientError as e:
            return f"AWS Error: {str(e)}"
        except Exception as e:
            return f"Error listing stacks: {str(e)}"

    def describe_stack(stack_name: str) -> str:
        """Get detailed information about a specific stack."""
        if not stack_name:
            return "Please provide a stack name."

        try:
            cf_client = client('cloudformation')
            response = cf_client.describe_stacks(StackName=stack_name)

            if not response.get('Stacks'):
                return f"Stack {stack_name} not found."

            stack = response['Stacks'][0]
            resources = cf_client.list_stack_resources(StackName=stack_name)

            details = [
                f"Stack: {stack['StackName']}",
                f"Status: {stack['StackStatus']}",
                f"Created: {stack['CreationTime']}",
                "\nResources:"
            ]

            for resource in resources.get('StackResourceSummaries', []):
                details.append(
                    f"- {resource['LogicalResourceId']}: "
                    f"{resource['ResourceType']} ({resource['ResourceStatus']})"
                )

            return "\n".join(details)
        except ClientError as e:
            return f"AWS Error: {str(e)}"
        except Exception as e:
            return f"Error describing stack: {str(e)}"

    def create_stack(stack_name: str, template_body: str) -> str:
        """Create a new CloudFormation stack."""
        if not stack_name or not template_body:
            return "Both stack name and template body are required."

        try:
            cf_client = client('cloudformation')
            response = cf_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
            )

            return f"Stack creation initiated. Stack ID: {response['StackId']}"
        except ClientError as e:
            return f"AWS Error: {str(e)}"
        except Exception as e:
            return f"Error creating stack: {str(e)}"

    def update_stack(stack_name: str, template_body: str) -> str:
        """Update an existing CloudFormation stack."""
        if not stack_name or not template_body:
            return "Both stack name and template body are required."

        try:
            cf_client = client('cloudformation')
            response = cf_client.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Capabilities=['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM']
            )

            return f"Stack update initiated. Stack ID: {response['StackId']}"
        except ClientError as e:
            if 'No updates are to be performed' in str(e):
                return "No updates needed - template matches current stack configuration."
            return f"AWS Error: {str(e)}"
        except Exception as e:
            return f"Error updating stack: {str(e)}"

    def delete_stack(stack_name: str) -> str:
        """Delete a CloudFormation stack."""
        if not stack_name:
            return "Please provide a stack name."

        try:
            cf_client = client('cloudformation')
            cf_client.delete_stack(StackName=stack_name)
            return f"Stack deletion initiated for {stack_name}"
        except ClientError as e:
            return f"AWS Error: {str(e)}"
        except Exception as e:
            return f"Error deleting stack: {str(e)}"

    def validate_template(template_body: str) -> str:
        """Validate a CloudFormation template."""
        if not template_body:
            return "Please provide a template body."

        try:
            cf_client = client('cloudformation')
            response = cf_client.validate_template(TemplateBody=template_body)

            validation_info = ["Template is valid."]
            if response.get('Parameters'):
                validation_info.append("\nRequired Parameters:")
                for param in response['Parameters']:
                    validation_info.append(
                        f"- {param['ParameterKey']}: "
                        f"{param.get('Description', 'No description')}"
                    )

            return "\n".join(validation_info)
        except ClientError as e:
            return f"Template validation failed: {str(e)}"
        except Exception as e:
            return f"Error validating template: {str(e)}"

    # Create tools with StructuredTool for proper argument handling
    return [
        StructuredTool.from_function(
            func=list_stacks,
            name="list_cf_stacks",
            description="List all active CloudFormation stacks in the AWS account"
        ),
        StructuredTool.from_function(
            func=describe_stack,
            name="describe_cf_stack",
            description="Get detailed information about a specific CloudFormation stack"
        ),
        StructuredTool.from_function(
            func=create_stack,
            name="create_cf_stack",
            description="Create a new CloudFormation stack"
        ),
        StructuredTool.from_function(
            func=update_stack,
            name="update_cf_stack",
            description="Update an existing CloudFormation stack"
        ),
        StructuredTool.from_function(
            func=delete_stack,
            name="delete_cf_stack",
            description="Delete a CloudFormation stack"
        ),
        StructuredTool.from_function(
            func=validate_template,
            name="validate_cf_template",
            description="Validate a CloudFormation template"
        )
    ]