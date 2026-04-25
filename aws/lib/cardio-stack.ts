import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import {
  DockerImageCode,
  DockerImageFunction,
  Architecture,
} from "aws-cdk-lib/aws-lambda";
import { Platform } from "aws-cdk-lib/aws-ecr-assets";
import {
  HttpApi,
  HttpMethod,
  CorsHttpMethod,
} from "aws-cdk-lib/aws-apigatewayv2";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations";
import { Rule, Schedule } from "aws-cdk-lib/aws-events";
import { LambdaFunction } from "aws-cdk-lib/aws-events-targets";
import { RetentionDays } from "aws-cdk-lib/aws-logs";
import { join } from "node:path";
import { IgnoreMode } from "aws-cdk-lib";

/**
 * Provisions the Cardio ML inference API on AWS.
 *
 * Resources:
 *   - Lambda (ARM64 container, 1024 MB, 30s timeout)
 *   - HTTP API (API Gateway v2) with throttling at 10 req/s burst 20
 *   - CloudWatch Events rule to keep the Lambda warm (every 5 min)
 */
export class CardioMlStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const projectRoot = join(__dirname, "..", "..");

    // ------------------------------------------------------------------
    // Lambda (ARM64 container built from the project root)
    // ------------------------------------------------------------------
    const fn = new DockerImageFunction(this, "CardioFn", {
      functionName: "cardio-ml-inference",
      code: DockerImageCode.fromImageAsset(projectRoot, {
        file: "aws/src/lambda/Dockerfile",
        platform: Platform.LINUX_ARM64,
        ignoreMode: IgnoreMode.DOCKER,
        exclude: [
          "aws/cdk.out",
          "aws/node_modules",
          "aws/dist",
          "venv",
          ".venv",
          ".git",
          "mlruns",
          "mlartifacts",
          "reports",
          "notebooks",
          "__pycache__",
          ".pytest_cache",
          ".ruff_cache",
          ".learning",
          ".remember",
          "*.egg-info",
        ],
      }),
      architecture: Architecture.ARM_64,
      memorySize: 1024,
      timeout: cdk.Duration.seconds(30),
      environment: {
        MODEL_PATH: "/var/task/model/pipeline.joblib",
        MODEL_VERSION: "1",
        CARDIO_N_JOBS: "1",
      },
      logRetention: RetentionDays.ONE_WEEK,
    });

    // ------------------------------------------------------------------
    // HTTP API with throttling
    // ------------------------------------------------------------------
    const integration = new HttpLambdaIntegration("CardioIntegration", fn);

    const api = new HttpApi(this, "CardioApi", {
      apiName: "cardio-ml-api",
      description: "Cardio ML inference API (project 26E1_3)",
      corsPreflight: {
        allowOrigins: ["*"],
        allowMethods: [CorsHttpMethod.GET, CorsHttpMethod.POST],
        allowHeaders: ["Content-Type"],
      },
    });

    // Catch-all route: FastAPI/Mangum handles internal routing.
    api.addRoutes({
      path: "/{proxy+}",
      methods: [HttpMethod.ANY],
      integration,
    });
    // Root route (API Gateway v2 doesn't include / in {proxy+}).
    api.addRoutes({
      path: "/",
      methods: [HttpMethod.ANY],
      integration,
    });

    // Throttling on the default stage via escape hatch (L2 doesn't expose this).
    const cfnStage = api.defaultStage?.node
      .defaultChild as cdk.aws_apigatewayv2.CfnStage;
    cfnStage.addPropertyOverride(
      "DefaultRouteSettings.ThrottlingBurstLimit",
      20,
    );
    cfnStage.addPropertyOverride(
      "DefaultRouteSettings.ThrottlingRateLimit",
      10,
    );

    // ------------------------------------------------------------------
    // Warming rule (every 10 min the handler responds and keeps the container warm)
    // ------------------------------------------------------------------
    const warmingRule = new Rule(this, "WarmingRule", {
      ruleName: "cardio-ml-warming",
      schedule: Schedule.rate(cdk.Duration.minutes(10)),
      description: "Keeps the Lambda warm to avoid cold starts.",
    });
    warmingRule.addTarget(new LambdaFunction(fn));

    // ------------------------------------------------------------------
    // Outputs
    // ------------------------------------------------------------------
    new cdk.CfnOutput(this, "ApiUrl", {
      value: api.url ?? "",
      description: "Base API URL. Append /health, /predict, /model-info, /ui.",
    });
    new cdk.CfnOutput(this, "HealthEndpoint", {
      value: `${api.url}health`,
      description: "Health check endpoint.",
    });
    new cdk.CfnOutput(this, "PredictEndpoint", {
      value: `${api.url}predict`,
      description: "Inference endpoint (POST).",
    });
    new cdk.CfnOutput(this, "SwaggerDocs", {
      value: `${api.url}docs`,
      description: "Interactive Swagger documentation.",
    });
  }
}
