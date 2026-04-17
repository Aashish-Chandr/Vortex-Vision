variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "vortexvision"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}
