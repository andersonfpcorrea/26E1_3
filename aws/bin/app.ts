#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { CardioMlStack } from "../lib/cardio-stack";

const app = new cdk.App();

new CardioMlStack(app, "CardioMlStack", {
  env: { region: "us-east-1" },
  description:
    "Cardiovascular disease screening inference API (project 26E1_3).",
});
