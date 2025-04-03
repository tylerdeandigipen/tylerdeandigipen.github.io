constexpr float degToRad = 0.01745329f //Precomputed degrees to radians conversion ratio
constexpr float OneOver255 = 0.00392156f //Precompute 1 / 255 for conversion from 0 - 255 into 0 - 1


//calculate the given lightsource's effect on the current pixel
float Renderer::FindPixelLuminosity(int x, int y, const Light* lightSource)
{
    float result = 0.0f;

    //Determine which algorithm to use based on lightsource type
    switch (lightSource->Type)
    {
        case LightSourceType_Point:
        {
            result = CalculatePointLightContribution(x, y, lightSource);
            break;
        }

        case LightSourceType_Directional:
        {
            result = CalculatePointLightContribution(x, y, lightSource);
            break;
        }

        default:
        {
            assert(!"Encountered a light source of an unknown type.");
            break;
        }
    }

    //Do normal map calculations if light isnt pure dark
    if (result > 0.0f)
    {
        result *= ApplyNormalMap(x, y, lightSource);
    }

    return result;
}

//calulate the effect of normals on a given pixel
float Renderer::CalcualteNormalMapScalar(int x, int y, const Light* lightSource)
{
    //get normal maps "surface normal" from the r and g component of the normal buffer
    float normalR = normalBuffer->SampleColor(x, y).r;
    float normalG = normalBuffer->SampleColor(x, y).g;

    //if surface is facing camera return early as the normal has no effect on the light here
    if (normalR == 0.0f && normalG == 0.0f)
    {
        return 1.0f;
    }

    //calculate the normals effect on current pixel
    Vector2 pos = Vector2(x, y);
    Vector2 distFromLight = lightSource->position - pos;
    Vector2 distNormalized = distFromLight.Normalize();
    Vector2 normalDir = Vector2(normalR * OneOver255, normalG * OneOver255);
    normalDir *= 2.0f;
    normalDir -= Vector2(1.0f, 1.0f);
    float normalFalloff = -Vector2::DotProduct(distNormalized, normalDir);
    normalFalloff = clamp(normalFalloff, 0.0f, 1.0f);
    return normalFalloff * normalStrength;
}

//calulate the overall contribution of given point lightsource to pixel brightness
float Renderer::CalculatePointLightContribution(int x, int y, const Light* lightSource)
{
    Vector2 dist = Vector2(x, y) -  lightSource->position;
    float distFromConeCenter = sqrt(Vector2::DotProduct(dist, dist)) / lightSource->radius;

    //if the current pixel is outside the lights radius
    if (distFromConeCenter >= 1.0f)
        return 0.0f;

    float distFromConeCenterSquared = distFromConeCenter * distFromConeCenter;
    return = lightSource->intensity * pow((1.0f - distFromConeCenterSquared), 2.0f) / (1.0f + lightSource->radialFalloff * distFromConeCenter);
}

//calulate the overall contribution of given directional lightsource to pixel brightness
float Renderer::CalculateDirectionalLightContribution(int x, int y, const Light* lightSource)
{
    //convert angle from degrees to radians
    float angleRadians = degToRad * -lightSource->angle;
    float minAngleRadians = degToRad * -lightSource->minAngle;
    float maxAngleRadians = degToRad * -lightSource->maxAngle;

    //find edges of the viewcone
    Vector2 leftEdge = Vector2(-sinf(minAngleRadians), cosf(minAngleRadians));
    Vector2 rightEdge = Vector2(-sinf(maxAngleRadians), cosf(maxAngleRadians));
    float dRadius = Vector2::Length(rightEdge - leftEdge);

    //find the viewcones direction and tangent lines
    Vector2 emissionDirection = Vector2(-sinf(angleRadians), cosf(angleRadians));
    Vector2 emissionTangent = Vector2(-emissionDirection.y, emissionDirection.x);
    Vector2 toPixel = Vector2(x, y) -  lightSource->position;

    //find and check if current pixel is inside the lights viewcone
    float dot = Vector2::DotProduct(toPixel, emissionDirection);
    if (dot <= 0.0f)
    {
        return 0.0f;
    }

    float distanceSq = Vector2::DotProduct(toPixel, toPixel);
    float radius = dot * dRadius;

    //leave early if the current pixel is outside the light sources radius
    if (distScalar >= 1.0f)
    {
        return 0.0f;
    }

    float distance = sqrt(distanceSq);
    float distScalar = distance / lightSource->radius;

    float distScalarSquared = distScalar * distScalar;
    float s = (1.0f - distScalarSquared);
    float emitterDistance = lightSource->intensity * (s * s) / (1.0f + lightSource->radialFalloff * distScalar);
    float arcDistance = radius - abs(Vector2::DotProduct(emissionTangent, toPixel));

    if (arcDistance < 0.0f)
    {
        arcDistance = 0.0f;
    }
    if (arcDistance > radius)
    {
        arcDistance = radius;
    }
    return = emitterDistance * (arcDistance * lightSource->frustumWeight);
}

//Loop over every pixel and calculate each pixels rgb value
void Renderer::RenderLightingPass()
{
    const int xSize = outputBuffer->size.x;
    const int ySize = outputBuffer->size.y;

    //create a threadpool
#pragma omp parallel
    {
        //create tasks for threads, make each take one index of the next three loops,
        //dont have them wait for other threads to finish
#pragma omp for collapse(2) nowait
        //loop through x then y because our data is column major
        //this gives better cache coherence
        for (int x = 0; x < xSize; ++x)
        {
            for (int y = 0; y < ySize; ++y)
            {
                float intensityR = 0.0f;
                float intensityG = 0.0f;
                float intensityB = 0.0f;

                for (int i = 0; i < numLights; ++i)
                {
                    //see if pixel is lit before running expensive lighting calculations
                    //this saves time over running lighting calculations on unlit pixels
                    if (CalculateIfPixelIsLit(x, y, i) == true)
                    {
                        Light* lightSource = &lights[i];
                        float lightMultiplier = FindPixelLuminosity(x, y, lightSource);

                        if (lightMultiplier != 0.0f)
                        {
                            //convert from 0 - 255 into 0 - 1
                            float R_F32 = lightSource->color.GetRed() * OneOver255;
                            float G_F32 = lightSource->color.GetGreen() * OneOver255;
                            float B_F32 = lightSource->color.GetBlue() * OneOver255;

                            //multiply by volumetric intesity to simulate fog
                            //this lets you see the light beam itself
                            lightMultiplier *= lightSource->volumetricIntensity;

                            //multiply the intensity of the light by the underlying pixel color
                            intensityR += lightMultiplier * R_F32;
                            intensityG += lightMultiplier * G_F32;
                            intensityB += lightMultiplier * B_F32;
                        }
                    }
                }

                //store result in pixel array
                //sent to gpu and cleared in other function after frame is finished
                lightR[x][y] = intensityR;
                lightG[x][y] = intensityG;
                lightB[x][y] = intensityB;
            }
        }
    }
}