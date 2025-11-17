output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.app.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.app.name
}

output "security_group_id" {
  description = "ID of the security group for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "task_definition_arn" {
  description = "ARN of the ECS task definition"
  value       = aws_ecs_task_definition.app.arn
}

output "instructions" {
  description = "Instructions for building and pushing the Docker image"
  value = <<-EOT
    To deploy your application:
    
    1. Authenticate Docker to ECR:
       aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.app.repository_url}
    
    2. Build the Docker image:
       docker build -t ${var.app_name} ./app
    
    3. Tag the image:
       docker tag ${var.app_name}:latest ${aws_ecr_repository.app.repository_url}:latest
    
    4. Push the image to ECR:
       docker push ${aws_ecr_repository.app.repository_url}:latest
    
    5. After pushing, the ECS service will automatically pull the new image.
       Find the public IP of your task in the ECS console and access your API at:
       http://<task-public-ip>:${var.container_port}
  EOT
}

