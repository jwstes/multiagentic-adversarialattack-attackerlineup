import time
import multiprocessing
from multiprocessing import Process, Queue


def info_agent_worker(result_queue):
    print("[Subprocess]: Info Agent process started.")
    from infoAgent import init, extractInfo
    model, processor = init()
    analysis = extractInfo(model, processor)
    print("[Subprocess]: Analysis complete. Putting result in queue.")
    result_queue.put(analysis)
    print("[Subprocess]: Process finished.")

def queryVectorStore_worker(result_queue, imageAnalysis):
    print("[Subprocess]: Query Vector Store process started.")
    from vectorstore.queryVectorStore import startQuery
    context = startQuery(imageAnalysis)
    print("[Subprocess]: Query Vector Store complete. Putting result in queue.")
    result_queue.put(context)
    print("[Subprocess]: Process finished.")

def thinking_agent_worker(result_queue, imageAnalysis, context):
    print("[Subprocess]: Thinking Agent process started.")
    from thinkingAgent import thinkAndSelectMethod
    results = thinkAndSelectMethod(imageAnalysis, context)
    print("[Subprocess]: Thinking complete. Putting result in queue.")
    result_queue.put(results)
    print("[Subprocess]: Process finished.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    ia_result_queue = Queue()
    qvc_result_queue = Queue()
    ta_result_queue = Queue()


    p = Process(target=info_agent_worker, args=(ia_result_queue,))
    print("[Main Process]: Starting Info Agent subprocess...")
    p.start()
    p.join()
    print("[Main Process]: Subprocess has finished.")
    imageAnalysis = ia_result_queue.get()
    print("[Main Process]: Received analysis result.")

    print("")
    print("=========================")
    print("")


    p = Process(target=queryVectorStore_worker, args=(qvc_result_queue,imageAnalysis))
    print("[Main Process]: Starting Query Vector Store subprocess...")
    p.start()
    p.join()
    print("[Main Process]: Subprocess has finished.")
    context = qvc_result_queue.get()
    print("[Main Process]: Received context result.")

    print("")
    print("=========================")
    print("")

    p = Process(target=thinking_agent_worker, args=(ta_result_queue,imageAnalysis,context))
    print("[Main Process]: Starting Thinking Agent subprocess...")
    p.start()
    p.join()
    print("[Main Process]: Subprocess has finished.")
    thinkingText = ta_result_queue.get()
    print("[Main Process]: Received result.")
    
    print("")
    print(thinkingText)