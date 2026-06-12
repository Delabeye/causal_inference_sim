
import zmq
import threading

class SwarmNetwork:
    """ 
    Classe qui permet de gérer le proxy notamment avec l'implémentation d'une boucle
    permettant de créer une large base de données 
    Corrige le bug de proxy entre deux runs
    """

    def __init__(self, port_in, port_out):
        self.port_in = port_in
        self.port_out = port_out
        self.proxy_ctx = zmq.Context()
        self.proxy_thread = None
        
        # "feu rouge" virtuel
        self.ready_event = threading.Event() 
        
        self.init_proxy()

    def init_proxy(self):
        def run_proxy():
            frontend = None
            backend = None

            for offset in range(0, 20, 2):
                test_in = self.port_in + offset
                test_out = self.port_out + offset

                try:
                    frontend = self.proxy_ctx.socket(zmq.XSUB)
                    frontend.setsockopt(zmq.LINGER, 0)
                    
                    backend = self.proxy_ctx.socket(zmq.XPUB)
                    backend.setsockopt(zmq.LINGER, 0)

                    frontend.bind(f"tcp://*:{test_in}")
                    backend.bind(f"tcp://*:{test_out}")

                    # On met à jour les variables de l'objet
                    self.port_in = test_in
                    self.port_out = test_out
                    print(f"[Swarm Network] Proxy démarré sur In: {self.port_in} -> Out: {self.port_out}")
                    
                    self.ready_event.set()
                    
                    zmq.proxy(frontend, backend)
                    break 

                except zmq.ZMQError:
                    if frontend: frontend.close()
                    if backend: backend.close()
                    continue 
                except zmq.ContextTerminated:
                    break
                except Exception as e:
                    print(f"[Swarm Network] Erreur inattendue : {e}")
                    break

            

            if frontend is not None:
                frontend.close()
            if backend is not None:
                backend.close()

        # On lance le thread
        self.proxy_thread = threading.Thread(target=run_proxy, daemon=True)
        self.proxy_thread.start()
        
        # Timeout de 5s pour éviter de bloquer indéfiniment si aucun port n'est libre.
        if not self.ready_event.wait(timeout=5.0):
            raise RuntimeError("Le proxy ZMQ n'a trouvé aucun port libre en moins de 5 secondes.")

    def stop_proxy(self):
        if self.proxy_ctx is not None:
           self.proxy_ctx.term()

        if self.proxy_thread is not None:
            self.proxy_thread.join(timeout=1.0)
