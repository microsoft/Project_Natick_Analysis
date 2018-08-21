# **Monitoring marine animals surrounding an underwater datacenter using Microsoft AI**
*Last updated August 21, 2018*

# **Introduction**
At Microsoft, we put our cloud and artificial intelligence (AI) tools in the hands of those working to solve global environmental challenges, through programs such as [AI for Earth](https://www.microsoft.com/en-us/aiforearth/). We also use these same tools to understand our own interaction with the environment, such as the work being done in concert with [Project Natick](https://natick.research.microsoft.com/).

[Project Natick](https://natick.research.microsoft.com/) seeks to understand the benefits and difficulties in deploying subsea datacenters worldwide; it is the world&#39;s first deployed underwater datacenter and it was designed with an emphasis on sustainability. Phase 2 extends the research accomplished in Phase 1 by deploying a full-scale datacenter module in the North Sea, powered by renewable energy. Project Natick uses AI to monitor the servers and other equipment for signs of failure and to identify any correlations between the environment and server longevity.

Because Project Natick operates like a standard land datacenter, the computers inside can be used for machine learning to provide AI to other applications, just as in any other Microsoft datacenter. We are also using AI to monitor the surrounding aquatic environment, as a first step to understanding what impact, if any, the datacenter may have.

## **Monitoring marine life using object detection**

The Project Natick datacenter is equipped with various sensors to monitor server conditions and the environment, including two underwater cameras, which are available as live video streams (check out the livestream on the [Project Natick homepage](https://natick.research.microsoft.com/#section-live)). These cameras allow us to monitor the surrounding environment from two fixed locations outside the datacenter in real time.

We want to count the marine life seen by the cameras. Manually counting the marine life in each frame in the video stream requires significant amount of effort. To solve this, we can leverage object detection to automate the monitoring and counting of marine life.

In each frame, we count the number of marine creatures. We model this as an object detection problem. Object detection combines the task of classification with localization, and outputs both a category and a set of coordinates representing the bounding box for each object detected in the image. 

# **How to Run**
Please go through the following steps to be able to run `natick_OD.py`, which will perform Object Detection on a Project Natick livestream and push the data to a Power BI dashboard. 

1. Clone this repository into a directory of your choice
2. Ensure you have all dependencies installed (see below)
3. Create a Power BI Streaming Dataset (see below)
4. Add your Power BI Streaming Dataset URL to line 139 of `natick_OD.py`
5. Run `python natick_OD.py`

## Dependencies
- `pip install cython`
- `pip install pillow`
- `pip install lxml`
- `pip install matplotlib`
- `pip install imutils`
- `pip install opencv-python`
- `pip install --ignore-installed --upgrade tensorflow`

## Creating a Power BI Streaming Dataset
Create a Power BI streaming dataset following [this tutorial](https://docs.microsoft.com/en-us/power-bi/service-real-time-streaming). When creating your dataset, add the following values.
![Power BI Streaming Dataset Values](images/PowerBIsetup.PNG)

## Running the Code
Then edit line 139 of `natick_OD.py` to use your Power BI Push URL. Finally, navigate to where you cloned the repo and run `python natick_OD.py`

## Note
This repo uses code from the [TensorFlow Object Detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection). We have edited the file `utils\visualization_utils.py` so that it displays the fish count in the bottom left corner of the video.

## Getting data
If you want to train from scratch, the annotated data is located under the [Release Tab](https://github.com/Microsoft/Project_Natick_Analysis/releases)

# Additional Information
Continue reading for additional information that is not necessary for running the code on your own machine.

## **Deploying the model to the Project Natick datacenter**

Another question we asked was - can we deploy the model to the Natick datacenter to monitor the wildlife teeming around the data center?

We chose to use CPUs to process the input videos and tested locally to make sure it works well. However, the default TensorFlow pre-built binary does not have optimizations such as AVX or FMA built-in to fully utilize modern CPUs. To better utilize the CPUs, we built the TensorFlow binary from source code, turning on all the optimization for Intel CPU by following Intel&#39;s [documentation](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide). With all the optimization, we can increase the processing speed by 50 percent from around two frame per second to three frame per second. The build command is like below:

```
bazel build --config=mkl -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mavx512f --copt=-mavx512pf --copt=-mavx512cd --copt=-mavx512er --copt="-DEIGEN_USE_VML"

//tensorflow/tools/pip_package:build_pip_package
```

## **Real-time environmental monitoring with Power BI**

Environmental scientists and aquatic scientists may benefit from a more intuitive way of monitoring the statistics of the underwater datacenter, such that they can quickly gain insight as to what is going on, through powerful visualization via Power BI.

Power BI has a notion of real-time datasets which provides the ability to accept streamed data and [update dashboards in real time](https://docs.microsoft.com/en-us/power-bi/service-real-time-streaming). It is intuitive to call the REST API to post data to the Power BI dashboard with a few lines of code:

```
# REST API endpoint, given to you when you create an API streaming dataset
# Will be of the format: https://api.powerbi.com/beta/<tenant id>/datasets/< dataset id>/rows?key=<key id>
REST_API_URL = ' *** Your Push API URL goes here *** '
# ensure that timestamp string is formatted properly
now = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%Z")
# data that we're sending to Power BI REST API
data = '[{{ "timestamp": "{0}", "fish_count": "{1}", "arrow_worm_count": "{2}" }}]'.format(now, fish_count, arrow_worm_count)
req = urllib2.Request(REST_API_URL, data)
response = urllib2.urlopen(req)
```

Because the animals may move quickly, we need to carefully balance between capturing data for many frames in short succession, sending to the Power BI dashboard, and consuming compute resources. We chose to push the analyzed data (for example, fish count) to Power BI three times per second to achieve this balance.

# **Summary**
Monitoring the environmental impact is an important topic, and AI can help make this process more scalable, and automated. In this post, we explained how we developed a deep learning solution for environment monitoring near the underwater data center. In this solution, we show how to ingest and store the data, and train an underwater animal detector to detect the marine life seen by the cameras. The model is then deployed to the machines in the data center to monitor the marine life. At the same time, we also explored how to analyze the video streams and leverage Power BI&#39;s streaming APIs to monitor the marine life over time.

If you have questions or comments, please leave a message here.

# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
